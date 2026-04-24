from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.summarize import summarize_csv


def test_summarize_csv_builds_group_summary(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {"task_type": "gsm8k", "selected_model": "cheap", "selected_budget": 0, "is_correct": 1, "cost_usd": 0.1, "latency_s": 1.0},
            {"task_type": "gsm8k", "selected_model": "cheap", "selected_budget": 0, "is_correct": 0, "cost_usd": 0.2, "latency_s": 2.0},
        ]
    ).to_csv(csv_path, index=False)

    summary = summarize_csv(str(csv_path))
    assert "accuracy_ci_lower" in summary.columns
    assert int(summary["n"].iloc[0]) == 2
