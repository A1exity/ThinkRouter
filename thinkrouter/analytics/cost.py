from __future__ import annotations

import pandas as pd


def summarize_costs(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_by = group_by or ["task_type", "selected_model", "selected_budget"]
    columns = [column for column in group_by if column in df.columns]
    if not columns:
        columns = ["task_type"] if "task_type" in df.columns else [df.columns[0]]
    summary = (
        df.groupby(columns, dropna=False)
        .agg(
            n=("cost_usd", "size"),
            avg_cost=("cost_usd", "mean"),
            total_cost=("cost_usd", "sum"),
            avg_tokens=("total_tokens", "mean"),
            total_tokens=("total_tokens", "sum"),
        )
        .reset_index()
    )
    return summary.sort_values(["avg_cost", "total_cost"], ascending=[True, True]).reset_index(drop=True)
