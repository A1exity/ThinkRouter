from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thinkrouter.app.router import make_feature_frame
from thinkrouter.experiments.evaluate_policy import UtilityWeights, add_sample_id, summarize_selection, utility

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
    return LearnedPolicyArtifact(
        pipeline=pipeline,
        weights=asdict(weights),
        label_counts=label_counts,
        training_rows=int(len(examples)),
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


def replay_learned_policy(artifact: LearnedPolicyArtifact, df: pd.DataFrame) -> pd.DataFrame:
    prepared = add_sample_id(df)
    predictions = predict_policy_budgets(artifact, prepared)
    selected_rows: list[pd.Series] = []
    for _, prediction in predictions.iterrows():
        group = prepared[
            (prepared["sample_id"].astype(str) == str(prediction["sample_id"]))
            & (prepared["selected_model"].astype(str) == str(prediction["selected_model"]))
        ].copy()
        exact = group[group["selected_budget"].astype(int) == int(prediction["predicted_budget"])]
        candidates = exact if not exact.empty else group.sort_values(["cost_usd", "latency_s", "selected_budget"], ascending=[True, True, True])
        selected = candidates.iloc[0].copy()
        selected["predicted_budget"] = int(prediction["predicted_budget"])
        selected["policy"] = "learned_policy"
        selected_rows.append(selected)
    return pd.DataFrame(selected_rows)


def evaluate_learned_policy(csv_path: str, model_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact = load_artifact(model_path)
    df = pd.read_csv(csv_path)
    selected = replay_learned_policy(artifact, df)
    summary = pd.DataFrame([summarize_selection("learned_policy", selected)])
    return summary, selected


def artifact_metadata(artifact: LearnedPolicyArtifact) -> dict[str, Any]:
    return {
        "training_rows": artifact.training_rows,
        "label_counts": artifact.label_counts,
        "weights": artifact.weights,
    }
