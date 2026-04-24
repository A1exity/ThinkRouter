from __future__ import annotations

import pandas as pd


def summarize_latency(df: pd.DataFrame, group_by: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    group_by = group_by or ["task_type", "selected_model", "selected_budget"]
    columns = [column for column in group_by if column in df.columns]
    if not columns:
        columns = ["task_type"] if "task_type" in df.columns else [df.columns[0]]
    summary = (
        df.groupby(columns, dropna=False)
        .agg(
            n=("latency_s", "size"),
            avg_latency=("latency_s", "mean"),
            p50_latency=("latency_s", "median"),
            p95_latency=("latency_s", lambda s: s.quantile(0.95)),
            max_latency=("latency_s", "max"),
        )
        .reset_index()
    )
    return summary.sort_values(["avg_latency", "p95_latency"], ascending=[True, True]).reset_index(drop=True)
