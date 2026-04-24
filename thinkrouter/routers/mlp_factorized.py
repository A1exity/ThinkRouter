from __future__ import annotations

from dataclasses import dataclass

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thinkrouter.adapters.base import ModelConfig
from thinkrouter.app.budgets import BUDGET_LEVELS
from thinkrouter.app.schemas import RouteDecision, TaskType
from thinkrouter.features import make_feature_frame
from thinkrouter.features.embedding import SEMANTIC_FEATURE_COLUMNS
from thinkrouter.routers.common import RouterWeights, clamp_budget, estimate_accuracy, estimate_cost, estimate_latency

NUMERIC_FEATURES = [
    "char_count",
    "word_count",
    "digit_count",
    "digit_density",
    "math_symbol_count",
    "punctuation_count",
    "code_marker_count",
    "avg_word_length",
    "cheap_probe_difficulty",
    "cheap_probe_confidence",
    "cheap_probe_consistency",
] + list(SEMANTIC_FEATURE_COLUMNS)
CATEGORICAL_FEATURES = ["task_type"]


@dataclass(frozen=True)
class FactorizedRouterArtifact:
    model_pipeline: Pipeline
    budget_pipeline: Pipeline
    fallback_model_id: str
    fallback_budget: int


def train_factorized_router(rows: list[dict[str, object]]) -> FactorizedRouterArtifact:
    if not rows:
        raise ValueError("Training rows are empty.")
    frame = make_feature_frame(rows)
    model_labels = [str(row["selected_model"]) for row in rows]
    budget_labels = [int(row["selected_budget"]) for row in rows]
    model_pipeline = _build_pipeline(model_labels)
    budget_pipeline = _build_pipeline(budget_labels)
    model_pipeline.fit(frame, model_labels)
    budget_pipeline.fit(frame, budget_labels)
    return FactorizedRouterArtifact(
        model_pipeline=model_pipeline,
        budget_pipeline=budget_pipeline,
        fallback_model_id=str(model_labels[0]),
        fallback_budget=int(budget_labels[0]),
    )


def _build_pipeline(labels) -> Pipeline:
    classifier = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=0)
    if len(set(labels)) == 1:
        classifier = DummyClassifier(strategy="most_frequent")
    return Pipeline(
        steps=[
            ("features", ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])),
            ("clf", classifier),
        ]
    )


class MLPFactorizedRouter:
    name = "mlp_factorized"

    def __init__(self, models: list[ModelConfig], artifact: FactorizedRouterArtifact | None = None, weights: RouterWeights | None = None) -> None:
        self.models = models
        self.artifact = artifact
        self.weights = weights or RouterWeights()

    def route(self, query: str, task_type: TaskType = "custom") -> RouteDecision:
        if self.artifact is None:
            from thinkrouter.routers.threshold import ThresholdRouter

            fallback = ThresholdRouter(self.models, self.weights).route(query, task_type)
            return fallback.model_copy(update={"router_name": self.name, "explanation": f"{fallback.explanation} Fell back to threshold: missing factorized artifact."})

        frame = make_feature_frame([{"query": query, "task_type": task_type}])
        model_id = str(self.artifact.model_pipeline.predict(frame)[0])
        budget = clamp_budget(int(self.artifact.budget_pipeline.predict(frame)[0]))
        model_conf = _top_probability(self.artifact.model_pipeline, frame)
        budget_conf = _top_probability(self.artifact.budget_pipeline, frame)
        confidence = min(model_conf, budget_conf)
        model = next((item for item in self.models if item.model_id == model_id), self.models[0])
        difficulty_score = float(frame.iloc[0]["cheap_probe_difficulty"])
        difficulty_label = "easy" if difficulty_score < 0.35 else "medium" if difficulty_score < 0.7 else "hard"
        return RouteDecision(
            model_id=model.model_id,
            budget=budget,
            difficulty=difficulty_label,
            estimated_accuracy=estimate_accuracy(model.tier, budget, difficulty_score, task_type),
            estimated_cost=estimate_cost(model.tier, budget),
            estimated_latency=estimate_latency(model.tier, budget, task_type),
            explanation=f"Factorized router selected model={model.model_id}, budget={budget}, confidence={confidence:.3f}.",
            route_confidence=confidence,
            router_name=self.name,
            fallback_triggered=False,
            fallback_reason=None,
        )


def _top_probability(pipeline: Pipeline, frame) -> float:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(frame)[0]
        return float(max(probabilities))
    return 0.5


def save_factorized_artifact(artifact: FactorizedRouterArtifact, out_path: str) -> None:
    joblib.dump(artifact, out_path)


def load_factorized_artifact(path: str) -> FactorizedRouterArtifact:
    artifact = joblib.load(path)
    if isinstance(artifact, FactorizedRouterArtifact):
        return artifact
    if isinstance(artifact, dict):
        return FactorizedRouterArtifact(**artifact)
    raise ValueError(f"Unsupported factorized artifact type: {type(artifact)!r}")
