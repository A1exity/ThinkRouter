from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.analyze_failures import parse_metadata
from thinkrouter.training.objectives import UtilityObjective, trace_utility


def add_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        metadata_values = out["metadata"] if "metadata" in out.columns else [None] * len(out)
        row_ids = out["id"] if "id" in out.columns else range(len(out))
        out["sample_id"] = [parse_metadata(value).get("sample_id", str(row_id)) for value, row_id in zip(metadata_values, row_ids)]
    return out


def derive_joint_examples(df: pd.DataFrame, objective: UtilityObjective | None = None) -> list[dict[str, object]]:
    objective = objective or UtilityObjective()
    prepared = add_sample_id(df)
    examples: list[dict[str, object]] = []
    for sample_id, group in prepared.groupby("sample_id", sort=False):
        ranked = group.copy()
        ranked["utility"] = [trace_utility(row, objective) for _, row in ranked.iterrows()]
        best = ranked.sort_values(["utility", "is_correct", "cost_usd", "latency_s"], ascending=[False, False, True, True]).iloc[0]
        examples.append(
            {
                "sample_id": str(sample_id),
                "query": str(best["query"]),
                "task_type": str(best["task_type"]),
                "selected_model": str(best["selected_model"]),
                "selected_budget": int(best["selected_budget"]),
            }
        )
    return examples


def derive_factorized_examples(df: pd.DataFrame, objective: UtilityObjective | None = None) -> list[dict[str, object]]:
    return derive_joint_examples(df, objective)
