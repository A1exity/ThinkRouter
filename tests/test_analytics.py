from __future__ import annotations

import pandas as pd

from thinkrouter.analytics import summarize_costs, summarize_latency


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"task_type": "gsm8k", "selected_model": "cheap", "selected_budget": 0, "cost_usd": 0.1, "latency_s": 1.0, "total_tokens": 10},
            {"task_type": "gsm8k", "selected_model": "cheap", "selected_budget": 0, "cost_usd": 0.2, "latency_s": 2.0, "total_tokens": 20},
            {"task_type": "gsm8k", "selected_model": "strong", "selected_budget": 256, "cost_usd": 0.3, "latency_s": 4.0, "total_tokens": 30},
        ]
    )


def test_summarize_costs_groups_cost_columns() -> None:
    summary = summarize_costs(_frame())
    assert set(summary.columns) >= {"avg_cost", "total_cost", "avg_tokens", "total_tokens"}
    assert summary["n"].sum() == 3


def test_summarize_latency_groups_latency_columns() -> None:
    summary = summarize_latency(_frame())
    assert set(summary.columns) >= {"avg_latency", "p50_latency", "p95_latency", "max_latency"}
    assert summary["n"].sum() == 3
