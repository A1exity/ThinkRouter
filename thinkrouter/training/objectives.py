from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UtilityObjective:
    alpha: float = 1.0
    beta: float = 5.0
    gamma: float = 0.02


def trace_utility(row: pd.Series, objective: UtilityObjective | None = None) -> float:
    objective = objective or UtilityObjective()
    return (
        objective.alpha * float(bool(row.get("is_correct", False)))
        - objective.beta * float(row.get("cost_usd", 0.0))
        - objective.gamma * float(row.get("latency_s", 0.0))
    )
