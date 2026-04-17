from __future__ import annotations

from dataclasses import dataclass

from thinkrouter.app.budgets import BUDGET_LEVELS
from thinkrouter.app.schemas import DifficultyLabel, RouteDecision


@dataclass(frozen=True)
class PolicyWeights:
    alpha: float = 1.0
    beta: float = 5.0
    gamma: float = 0.02


class HeuristicDifficultyEstimator:
    def predict(self, query: str, task_type: str = "custom") -> DifficultyLabel:
        text = query.lower()
        length = len(query.split())
        digit_count = sum(ch.isdigit() for ch in query)
        math_symbols = sum(ch in "+-*/=^" for ch in query)
        code_markers = sum(marker in text for marker in ["def ", "class ", "return", "assert", "```"])
        if task_type == "humaneval" or code_markers >= 2 or length > 120:
            return "hard"
        if task_type == "math" or digit_count >= 8 or math_symbols >= 4 or length > 55:
            return "medium"
        return "easy"


class JointPolicyEngine:
    def __init__(self, model_ids: list[str], weights: PolicyWeights | None = None) -> None:
        if not model_ids:
            raise ValueError("JointPolicyEngine requires at least one model_id.")
        self.model_ids = model_ids
        self.weights = weights or PolicyWeights()
        self.difficulty_estimator = HeuristicDifficultyEstimator()

    def route(self, query: str, task_type: str = "custom") -> RouteDecision:
        difficulty = self.difficulty_estimator.predict(query, task_type)
        best: RouteDecision | None = None
        best_utility = float("-inf")
        for model_index, model_id in enumerate(self.model_ids):
            for budget in BUDGET_LEVELS:
                accuracy = self._estimate_accuracy(model_index, budget, difficulty, task_type)
                cost = self._estimate_cost(model_index, budget)
                latency = self._estimate_latency(model_index, budget)
                utility = self.weights.alpha * accuracy - self.weights.beta * cost - self.weights.gamma * latency
                if utility > best_utility:
                    best_utility = utility
                    best = RouteDecision(
                        model_id=model_id,
                        budget=budget,
                        difficulty=difficulty,
                        estimated_accuracy=accuracy,
                        estimated_cost=cost,
                        estimated_latency=latency,
                        explanation=(
                            f"Selected by heuristic utility={utility:.3f}; "
                            f"difficulty={difficulty}, model_rank={model_index}, budget={budget}."
                        ),
                    )
        assert best is not None
        return best

    def _estimate_accuracy(self, model_index: int, budget: int, difficulty: DifficultyLabel, task_type: str) -> float:
        base_by_model = [0.45, 0.65, 0.78]
        base = base_by_model[min(model_index, len(base_by_model) - 1)]
        budget_bonus = {0: 0.0, 256: 0.05, 1024: 0.10, 4096: 0.13}[budget]
        difficulty_penalty = {"easy": 0.0, "medium": 0.12, "hard": 0.22}[difficulty]
        if task_type == "humaneval":
            difficulty_penalty += 0.05
        return max(0.0, min(0.98, base + budget_bonus - difficulty_penalty))

    def _estimate_cost(self, model_index: int, budget: int) -> float:
        model_factor = [0.0, 0.0003, 0.0008][min(model_index, 2)]
        return model_factor * (1 + budget / 1024)

    def _estimate_latency(self, model_index: int, budget: int) -> float:
        return 0.4 + model_index * 0.8 + budget / 1500
