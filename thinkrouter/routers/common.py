from __future__ import annotations

from dataclasses import dataclass

from thinkrouter.app.budgets import BUDGET_LEVELS


@dataclass(frozen=True)
class RouterWeights:
    alpha: float = 1.0
    beta: float = 5.0
    gamma: float = 0.02


def estimate_cost(tier: str | None, budget: int) -> float:
    base = {"cheap": 0.00025, "mid": 0.0012, "strong": 0.004}.get(tier or "", 0.0008)
    return base * (1 + budget / 1024)


def estimate_latency(tier: str | None, budget: int, task_type: str) -> float:
    tier_base = {"cheap": 4.0, "mid": 9.0, "strong": 3.0}.get(tier or "", 5.0)
    task_multiplier = {"gsm8k": 1.0, "math": 1.2, "humaneval": 1.6}.get(task_type, 1.0)
    return (tier_base + budget / 180.0) * task_multiplier


def estimate_accuracy(tier: str | None, budget: int, difficulty_score: float, task_type: str) -> float:
    tier_base = {"cheap": 0.72, "mid": 0.84, "strong": 0.9}.get(tier or "", 0.78)
    budget_bonus = {0: 0.0, 256: 0.04, 1024: 0.08, 4096: 0.1}.get(int(budget), 0.08)
    task_penalty = {"gsm8k": 0.0, "math": 0.05, "humaneval": 0.1}.get(task_type, 0.02)
    return max(0.05, min(0.98, tier_base + budget_bonus - difficulty_score * 0.25 - task_penalty))


def clamp_budget(value: int) -> int:
    if value in BUDGET_LEVELS:
        return value
    return min(BUDGET_LEVELS, key=lambda budget: abs(budget - int(value)))
