from __future__ import annotations

from thinkrouter.adapters.base import ModelConfig
from thinkrouter.app.budgets import BUDGET_LEVELS
from thinkrouter.app.schemas import RouteDecision, TaskType
from thinkrouter.features import extract_query_features
from thinkrouter.routers.common import RouterWeights, estimate_accuracy, estimate_cost, estimate_latency


class ThresholdRouter:
    name = "threshold"

    def __init__(self, models: list[ModelConfig], weights: RouterWeights | None = None) -> None:
        self.models = models
        self.weights = weights or RouterWeights()

    def route(self, query: str, task_type: TaskType = "custom") -> RouteDecision:
        features = extract_query_features(query, task_type)
        difficulty_score = float(features["cheap_probe_difficulty"])
        target_tier = "cheap"
        if difficulty_score >= 0.75 or task_type == "humaneval":
            target_tier = "strong"
        elif difficulty_score >= 0.45 or task_type == "math":
            target_tier = "mid"

        target_budget = 0
        if difficulty_score >= 0.8:
            target_budget = 1024
        elif difficulty_score >= 0.4:
            target_budget = 256

        best_model = self._pick_model(target_tier)
        accuracy = estimate_accuracy(best_model.tier, target_budget, difficulty_score, task_type)
        cost = estimate_cost(best_model.tier, target_budget)
        latency = estimate_latency(best_model.tier, target_budget, task_type)
        confidence = max(0.15, 1.0 - abs(difficulty_score - 0.5))
        difficulty_label = "easy" if difficulty_score < 0.35 else "medium" if difficulty_score < 0.7 else "hard"
        return RouteDecision(
            model_id=best_model.model_id,
            budget=target_budget,
            difficulty=difficulty_label,
            estimated_accuracy=accuracy,
            estimated_cost=cost,
            estimated_latency=latency,
            explanation=f"Threshold router selected tier={best_model.tier}, budget={target_budget}, score={difficulty_score:.3f}.",
            route_confidence=confidence,
            router_name=self.name,
            fallback_triggered=False,
            fallback_reason=None,
        )

    def _pick_model(self, tier: str) -> ModelConfig:
        for model in self.models:
            if model.tier == tier:
                return model
        if tier == "mid":
            for candidate_tier in ("strong", "cheap"):
                for model in self.models:
                    if model.tier == candidate_tier:
                        return model
        return self.models[min(len(self.models) - 1, 0 if tier == "cheap" else len(self.models) - 1)]
