from __future__ import annotations

from thinkrouter.adapters.base import ModelConfig
from thinkrouter.app.schemas import RouteDecision, TaskType
from thinkrouter.routers.common import RouterWeights
from thinkrouter.routers.mlp_factorized import FactorizedRouterArtifact, MLPFactorizedRouter
from thinkrouter.routers.threshold import ThresholdRouter


class UncertaintyAwareRouter:
    name = "uncertainty_aware"

    def __init__(
        self,
        models: list[ModelConfig],
        artifact: FactorizedRouterArtifact | None = None,
        weights: RouterWeights | None = None,
        confidence_threshold: float = 0.55,
    ) -> None:
        self.models = models
        self.weights = weights or RouterWeights()
        self.primary = MLPFactorizedRouter(models, artifact=artifact, weights=self.weights)
        self.fallback = ThresholdRouter(models, self.weights)
        self.confidence_threshold = confidence_threshold

    def route(self, query: str, task_type: TaskType = "custom") -> RouteDecision:
        primary = self.primary.route(query, task_type)
        missing_artifact = "missing factorized artifact" in primary.explanation.lower()
        if missing_artifact:
            fallback = self.fallback.route(query, task_type)
            return fallback.model_copy(
                update={
                    "router_name": self.name,
                    "fallback_triggered": True,
                    "fallback_reason": "missing_artifact",
                    "route_confidence": primary.route_confidence,
                    "explanation": f"Uncertainty-aware fallback due to missing factorized artifact. {fallback.explanation}",
                }
            )
        if primary.route_confidence is not None and primary.route_confidence >= self.confidence_threshold:
            return primary.model_copy(update={"router_name": self.name})
        fallback = self.fallback.route(query, task_type)
        return fallback.model_copy(
            update={
                "router_name": self.name,
                "fallback_triggered": True,
                "fallback_reason": "low_confidence",
                "route_confidence": primary.route_confidence,
                "explanation": f"Uncertainty-aware fallback after primary confidence={primary.route_confidence or 0.0:.3f}. {fallback.explanation}",
            }
        )
