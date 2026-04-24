from __future__ import annotations

from dataclasses import dataclass

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from thinkrouter.adapters.base import ModelConfig
from thinkrouter.app.budgets import BUDGET_LEVELS
from thinkrouter.app.schemas import RouteDecision, TaskType
from thinkrouter.features import make_feature_frame
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
    "embedding_hash_mean",
    "embedding_hash_std",
    "semantic_bucket",
    "semantic_diversity",
    "cheap_probe_difficulty",
    "cheap_probe_confidence",
    "cheap_probe_consistency",
]
CATEGORICAL_FEATURES = ["task_type"]


@dataclass(frozen=True)
class LogRegJointArtifact:
    pipeline: Pipeline
    labels: list[str]


def train_logreg_joint_router(rows: list[dict[str, object]]) -> LogRegJointArtifact:
    if not rows:
        raise ValueError("Training rows are empty.")
    labels = [f"{row['selected_model']}::{int(row['selected_budget'])}" for row in rows]
    classifier = LogisticRegression(max_iter=500, class_weight="balanced", random_state=0)
    if len(set(labels)) == 1:
        classifier = DummyClassifier(strategy="most_frequent")
    pipeline = Pipeline(
        steps=[
            ("features", ColumnTransformer([("num", StandardScaler(), NUMERIC_FEATURES), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)])),
            ("clf", classifier),
        ]
    )
    pipeline.fit(make_feature_frame(rows), labels)
    return LogRegJointArtifact(pipeline=pipeline, labels=sorted(set(labels)))


class LogRegJointRouter:
    name = "logreg_joint"

    def __init__(self, models: list[ModelConfig], artifact: LogRegJointArtifact | None = None, weights: RouterWeights | None = None) -> None:
        self.models = models
        self.artifact = artifact
        self.weights = weights or RouterWeights()

    def route(self, query: str, task_type: TaskType = "custom") -> RouteDecision:
        if self.artifact is None:
            from thinkrouter.routers.threshold import ThresholdRouter

            fallback = ThresholdRouter(self.models, self.weights).route(query, task_type)
            return fallback.model_copy(update={"router_name": self.name, "explanation": f"{fallback.explanation} Fell back to threshold: missing artifact."})

        frame = make_feature_frame([{"query": query, "task_type": task_type}])
        prediction = str(self.artifact.pipeline.predict(frame)[0])
        proba = self.artifact.pipeline.predict_proba(frame)[0] if hasattr(self.artifact.pipeline, "predict_proba") else None
        confidence = float(max(proba)) if proba is not None else 0.5
        model_id, budget_text = prediction.split("::", 1)
        budget = clamp_budget(int(budget_text))
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
            explanation=f"LogReg joint router selected {prediction} with confidence={confidence:.3f}.",
            route_confidence=confidence,
            router_name=self.name,
            fallback_triggered=False,
            fallback_reason=None,
        )


def save_logreg_joint_artifact(artifact: LogRegJointArtifact, out_path: str) -> None:
    joblib.dump(artifact, out_path)


def load_logreg_joint_artifact(path: str) -> LogRegJointArtifact:
    artifact = joblib.load(path)
    if isinstance(artifact, LogRegJointArtifact):
        return artifact
    if isinstance(artifact, dict):
        return LogRegJointArtifact(**artifact)
    raise ValueError(f"Unsupported joint artifact type: {type(artifact)!r}")
