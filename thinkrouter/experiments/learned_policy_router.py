from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thinkrouter.app.router import make_feature_frame
from thinkrouter.experiments.evaluate_policy import UtilityWeights, add_sample_id, aggregate_utility_policy, fixed_budget_policy, summarize_selection, utility

NUMERIC_FEATURES = [
    "char_count",
    "word_count",
    "digit_count",
    "digit_density",
    "math_symbol_count",
    "punctuation_count",
    "code_marker_count",
    "avg_word_length",
]
CATEGORICAL_FEATURES = ["task_type", "selected_model"]


@dataclass(frozen=True)
class LearnedPolicyArtifact:
    pipeline: Pipeline
    weights: dict[str, float]
    label_counts: dict[int, int]
    training_rows: int
    fallback_budget: int | None = None
    safe_mode: bool = True
    diagnostics: dict[str, float | int | str] | None = None


def trace_utility(row: pd.Series, weights: UtilityWeights) -> float:
    return utility(
        pd.Series(
            {
                "accuracy": float(bool(row.get("is_correct", False))),
                "avg_cost": float(row.get("cost_usd", 0.0)),
                "avg_latency": float(row.get("latency_s", 0.0)),
            }
        ),
        weights,
    )


def derive_policy_training_examples(df: pd.DataFrame, weights: UtilityWeights | None = None) -> pd.DataFrame:
    weights = weights or UtilityWeights()
    required = {"query", "task_type", "selected_model", "selected_budget", "is_correct", "cost_usd", "latency_s"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Trace CSV is empty.")

    prepared = add_sample_id(df)
    rows: list[dict[str, object]] = []
    for (_, selected_model), group in prepared.groupby(["sample_id", "selected_model"], sort=False):
        ranked = group.copy()
        ranked["policy_utility"] = [trace_utility(row, weights) for _, row in ranked.iterrows()]
        ranked = ranked.sort_values(
            by=["policy_utility", "is_correct", "cost_usd", "latency_s", "selected_budget"],
            ascending=[False, False, True, True, True],
        )
        best = ranked.iloc[0]
        rows.append(
            {
                "sample_id": str(best["sample_id"]),
                "query": str(best["query"]),
                "task_type": str(best["task_type"]),
                "selected_model": str(selected_model),
                "selected_budget": int(best["selected_budget"]),
                "target_utility": float(best["policy_utility"]),
            }
        )
    return pd.DataFrame(rows)


def build_policy_pipeline(labels: pd.Series) -> Pipeline:
    classifier = LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)
    if labels.nunique() == 1:
        classifier = DummyClassifier(strategy="most_frequent")
    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC_FEATURES),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
                    ]
                ),
            ),
            ("clf", classifier),
        ]
    )


def train_learned_policy(csv_path: str, weights: UtilityWeights | None = None) -> LearnedPolicyArtifact:
    weights = weights or UtilityWeights()
    df = pd.read_csv(csv_path)
    examples = derive_policy_training_examples(df, weights)
    labels = examples["selected_budget"].astype(int)
    pipeline = build_policy_pipeline(labels)
    feature_frame = make_feature_frame(examples.to_dict("records"))
    pipeline.fit(feature_frame, labels)
    label_counts = {int(label): int(count) for label, count in labels.value_counts().sort_index().items()}
    fallback_budget, diagnostics = choose_safe_fallback_budget(df, weights)
    return LearnedPolicyArtifact(
        pipeline=pipeline,
        weights=asdict(weights),
        label_counts=label_counts,
        training_rows=int(len(examples)),
        fallback_budget=fallback_budget,
        safe_mode=True,
        diagnostics=diagnostics,
    )


def save_artifact(artifact: LearnedPolicyArtifact, out_path: str) -> None:
    joblib.dump(artifact, out_path)


def load_artifact(model_path: str) -> LearnedPolicyArtifact:
    artifact = joblib.load(model_path)
    if isinstance(artifact, LearnedPolicyArtifact):
        return artifact
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return LearnedPolicyArtifact(**artifact)
    raise ValueError(f"Unsupported learned policy artifact: {type(artifact)!r}")


def predict_policy_budgets(artifact: LearnedPolicyArtifact, df: pd.DataFrame) -> pd.DataFrame:
    prepared = add_sample_id(df)
    rows: list[dict[str, object]] = []
    for (sample_id, selected_model), group in prepared.groupby(["sample_id", "selected_model"], sort=False):
        first = group.iloc[0]
        rows.append(
            {
                "sample_id": str(sample_id),
                "query": str(first["query"]),
                "task_type": str(first["task_type"]),
                "selected_model": str(selected_model),
            }
        )
    examples = pd.DataFrame(rows)
    if examples.empty:
        return examples.assign(predicted_budget=pd.Series(dtype=int))
    feature_frame = make_feature_frame(examples.to_dict("records"))
    examples["predicted_budget"] = artifact.pipeline.predict(feature_frame).astype(int)
    return examples


def choose_safe_fallback_budget(df: pd.DataFrame, weights: UtilityWeights) -> tuple[int, dict[str, float | int | str]]:
    prepared = add_sample_id(df)
    aggregate_selected, aggregate_stats = aggregate_utility_policy(prepared, weights)
    fallback_budget = int(aggregate_selected["selected_budget"].iloc[0])
    fallback_summary = summarize_selection(f"fixed_budget_{fallback_budget}", aggregate_selected)
    oracle_examples = derive_policy_training_examples(prepared, weights)
    label_count = int(oracle_examples["selected_budget"].nunique())
    return (
        fallback_budget,
        {
            "safe_policy": "fallback_to_aggregate_utility",
            "fallback_budget": fallback_budget,
            "fallback_accuracy": float(fallback_summary["accuracy"]),
            "fallback_avg_cost": float(fallback_summary["avg_cost"]),
            "fallback_avg_latency": float(fallback_summary["avg_latency"]),
            "candidate_budget_count": label_count,
            "aggregate_utility": float(aggregate_stats.loc[aggregate_stats["selected_budget"].astype(int) == fallback_budget, "utility"].iloc[0]),
        },
    )


def calibrate_policy_artifact(artifact: LearnedPolicyArtifact, calibration_csv_path: str) -> tuple[LearnedPolicyArtifact, pd.DataFrame]:
    weights = UtilityWeights(**artifact.weights)
    df = add_sample_id(pd.read_csv(calibration_csv_path))
    rows: list[dict[str, Any]] = []

    raw_selected = replay_learned_policy(artifact, df, safe=False)
    rows.append(summarize_selection("learned_policy_raw", raw_selected))
    for budget in sorted(df["selected_budget"].astype(int).unique()):
        rows.append(summarize_selection(f"fixed_budget_{budget}", fixed_budget_policy(df, budget)))

    summary = pd.DataFrame(rows)
    summary["utility"] = [utility(row, weights) for _, row in summary.iterrows()]
    best = summary.sort_values(["utility", "accuracy", "avg_cost"], ascending=[False, False, True]).iloc[0]
    policy_name = str(best["policy"])
    if policy_name.startswith("fixed_budget_"):
        fallback_budget = int(policy_name.removeprefix("fixed_budget_"))
        calibrated = replace(
            artifact,
            fallback_budget=fallback_budget,
            safe_mode=True,
            diagnostics={
                **(artifact.diagnostics or {}),
                "calibration_policy": "fallback_to_calibrated_fixed_budget",
                "calibration_fallback_budget": fallback_budget,
                "calibration_utility": float(best["utility"]),
                "calibration_accuracy": float(best["accuracy"]),
                "calibration_avg_cost": float(best["avg_cost"]),
                "calibration_avg_latency": float(best["avg_latency"]),
            },
        )
    else:
        calibrated = replace(
            artifact,
            fallback_budget=None,
            safe_mode=False,
            diagnostics={
                **(artifact.diagnostics or {}),
                "calibration_policy": "raw_learned_policy",
                "calibration_utility": float(best["utility"]),
                "calibration_accuracy": float(best["accuracy"]),
                "calibration_avg_cost": float(best["avg_cost"]),
                "calibration_avg_latency": float(best["avg_latency"]),
            },
        )
    return calibrated, summary


def replay_learned_policy(artifact: LearnedPolicyArtifact, df: pd.DataFrame, safe: bool = True) -> pd.DataFrame:
    prepared = add_sample_id(df)
    predictions = predict_policy_budgets(artifact, prepared)
    selected_rows: list[pd.Series] = []
    for _, prediction in predictions.iterrows():
        group = prepared[
            (prepared["sample_id"].astype(str) == str(prediction["sample_id"]))
            & (prepared["selected_model"].astype(str) == str(prediction["selected_model"]))
        ].copy()
        predicted_budget = int(prediction["predicted_budget"])
        selected_budget = int(artifact.fallback_budget) if safe and artifact.safe_mode and artifact.fallback_budget is not None else predicted_budget
        exact = group[group["selected_budget"].astype(int) == selected_budget]
        candidates = exact if not exact.empty else group.sort_values(["cost_usd", "latency_s", "selected_budget"], ascending=[True, True, True])
        selected = candidates.iloc[0].copy()
        selected["predicted_budget"] = predicted_budget
        selected["policy_budget"] = int(selected["selected_budget"])
        selected["policy"] = "safe_learned_policy" if safe else "learned_policy"
        selected["safe_fallback_budget"] = artifact.fallback_budget
        selected_rows.append(selected)
    return pd.DataFrame(selected_rows)


def evaluate_learned_policy(csv_path: str, model_path: str, safe: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact = load_artifact(model_path)
    df = pd.read_csv(csv_path)
    selected = replay_learned_policy(artifact, df, safe=safe)
    if safe and artifact.safe_mode and artifact.fallback_budget is not None:
        name = f"safe_learned_policy_fallback_budget_{artifact.fallback_budget}"
    else:
        name = "learned_policy"
    summary = pd.DataFrame([summarize_selection(name, selected)])
    return summary, selected


def artifact_metadata(artifact: LearnedPolicyArtifact) -> dict[str, Any]:
    return {
        "training_rows": artifact.training_rows,
        "label_counts": artifact.label_counts,
        "weights": artifact.weights,
        "fallback_budget": artifact.fallback_budget,
        "safe_mode": artifact.safe_mode,
        "diagnostics": artifact.diagnostics,
    }
