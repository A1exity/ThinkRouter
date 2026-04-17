from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import pandas as pd

from thinkrouter.app.budgets import BUDGET_LEVELS
from thinkrouter.app.schemas import DifficultyLabel, RouteDecision, TaskType


@dataclass(frozen=True)
class PolicyWeights:
    alpha: float = 1.0
    beta: float = 5.0
    gamma: float = 0.02
    budget_hint_penalty: float = 0.03


class DifficultyEstimator(Protocol):
    def predict(self, query: str, task_type: str = "custom") -> DifficultyLabel:
        ...


class BudgetPredictor(Protocol):
    def predict(self, query: str, task_type: str, model_id: str, difficulty: DifficultyLabel) -> int:
        ...


def extract_query_features(query: str, task_type: str = "custom") -> dict[str, float | str]:
    text = query.lower()
    words = query.split()
    char_count = len(query)
    word_count = len(words)
    digit_count = sum(ch.isdigit() for ch in query)
    math_symbol_count = sum(ch in "+-*/=^" for ch in query)
    punctuation_count = sum(ch in ",.;:!?()[]{}" for ch in query)
    code_marker_count = sum(marker in text for marker in ["def ", "class ", "return", "assert", "```", "import "])
    digit_density = digit_count / max(char_count, 1)
    avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
    return {
        "query": query,
        "task_type": task_type,
        "char_count": float(char_count),
        "word_count": float(word_count),
        "digit_count": float(digit_count),
        "digit_density": float(digit_density),
        "math_symbol_count": float(math_symbol_count),
        "punctuation_count": float(punctuation_count),
        "code_marker_count": float(code_marker_count),
        "avg_word_length": float(avg_word_length),
    }


def make_feature_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    feature_rows: list[dict[str, object]] = []
    for row in rows:
        query = str(row.get("query", ""))
        task_type = str(row.get("task_type", "custom"))
        features = extract_query_features(query, task_type)
        for key, value in row.items():
            if key not in features:
                features[key] = value
        feature_rows.append(features)
    return pd.DataFrame(feature_rows)


class HeuristicDifficultyEstimator:
    def predict(self, query: str, task_type: str = "custom") -> DifficultyLabel:
        features = extract_query_features(query, task_type)
        length = int(features["word_count"])
        digit_count = int(features["digit_count"])
        math_symbols = int(features["math_symbol_count"])
        code_markers = int(features["code_marker_count"])
        if task_type == "humaneval" or code_markers >= 2 or length > 120:
            return "hard"
        if task_type == "math" or digit_count >= 8 or math_symbols >= 4 or length > 55:
            return "medium"
        return "easy"


class SklearnDifficultyEstimator:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

    def predict(self, query: str, task_type: str = "custom") -> DifficultyLabel:
        frame = make_feature_frame([{"query": query, "task_type": task_type}])
        prediction = str(self.pipeline.predict(frame)[0])
        if prediction not in {"easy", "medium", "hard"}:
            return "medium"
        return prediction  # type: ignore[return-value]


class SklearnBudgetPredictor:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self.pipeline = joblib.load(self.model_path)

    def predict(self, query: str, task_type: str, model_id: str, difficulty: DifficultyLabel) -> int:
        frame = make_feature_frame(
            [
                {
                    "query": query,
                    "task_type": task_type,
                    "selected_model": model_id,
                    "difficulty": difficulty,
                }
            ]
        )
        prediction = int(self.pipeline.predict(frame)[0])
        if prediction not in BUDGET_LEVELS:
            return 1024
        return prediction


class JointPolicyEngine:
    def __init__(
        self,
        model_ids: list[str],
        weights: PolicyWeights | None = None,
        difficulty_estimator: DifficultyEstimator | None = None,
        budget_predictor: BudgetPredictor | None = None,
    ) -> None:
        if not model_ids:
            raise ValueError("JointPolicyEngine requires at least one model_id.")
        self.model_ids = model_ids
        self.weights = weights or PolicyWeights()
        self.difficulty_estimator = difficulty_estimator or self._load_difficulty_estimator()
        self.budget_predictor = budget_predictor or self._load_budget_predictor()

    def route(self, query: str, task_type: TaskType = "custom") -> RouteDecision:
        difficulty = self.difficulty_estimator.predict(query, task_type)
        budget_hints = self._predict_budget_hints(query, task_type, difficulty)
        best: RouteDecision | None = None
        best_utility = float("-inf")
        for model_index, model_id in enumerate(self.model_ids):
            budget_hint = budget_hints.get(model_id)
            for budget in BUDGET_LEVELS:
                accuracy = self._estimate_accuracy(model_index, budget, difficulty, task_type)
                cost = self._estimate_cost(model_index, budget)
                latency = self._estimate_latency(model_index, budget)
                utility = self.weights.alpha * accuracy - self.weights.beta * cost - self.weights.gamma * latency
                if budget_hint is not None:
                    utility -= self._budget_distance_penalty(budget, budget_hint)
                if utility > best_utility:
                    best_utility = utility
                    hint_text = f", budget_hint={budget_hint}" if budget_hint is not None else ""
                    best = RouteDecision(
                        model_id=model_id,
                        budget=budget,
                        difficulty=difficulty,
                        estimated_accuracy=accuracy,
                        estimated_cost=cost,
                        estimated_latency=latency,
                        explanation=(
                            f"Selected by utility={utility:.3f}; difficulty={difficulty}, "
                            f"model_rank={model_index}, budget={budget}{hint_text}."
                        ),
                    )
        assert best is not None
        return best

    def _predict_budget_hints(self, query: str, task_type: str, difficulty: DifficultyLabel) -> dict[str, int]:
        if self.budget_predictor is None:
            return {}
        hints: dict[str, int] = {}
        for model_id in self.model_ids:
            hints[model_id] = self.budget_predictor.predict(query, task_type, model_id, difficulty)
        return hints

    def _budget_distance_penalty(self, budget: int, budget_hint: int) -> float:
        budget_index = BUDGET_LEVELS.index(budget)
        hint_index = BUDGET_LEVELS.index(budget_hint)
        return abs(budget_index - hint_index) * self.weights.budget_hint_penalty

    def _load_difficulty_estimator(self) -> DifficultyEstimator:
        model_path = os.getenv("THINKROUTER_DIFFICULTY_MODEL_PATH")
        if model_path and Path(model_path).exists():
            return SklearnDifficultyEstimator(model_path)
        return HeuristicDifficultyEstimator()

    def _load_budget_predictor(self) -> BudgetPredictor | None:
        model_path = os.getenv("THINKROUTER_BUDGET_MODEL_PATH")
        if model_path and Path(model_path).exists():
            return SklearnBudgetPredictor(model_path)
        return None

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
