from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.eval_baselines import summarize_baselines


def test_summarize_baselines_includes_policy_families(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "query": "q1",
                "task_type": "gsm8k",
                "selected_model": "cheap",
                "selected_budget": 0,
                "is_correct": True,
                "cost_usd": 0.01,
                "latency_s": 1.0,
                "metadata": "{'sample_id': 's1'}",
            },
            {
                "id": 2,
                "query": "q1",
                "task_type": "gsm8k",
                "selected_model": "strong",
                "selected_budget": 1024,
                "is_correct": True,
                "cost_usd": 0.05,
                "latency_s": 2.0,
                "metadata": "{'sample_id': 's1'}",
            },
            {
                "id": 3,
                "query": "q2",
                "task_type": "gsm8k",
                "selected_model": "cheap",
                "selected_budget": 0,
                "is_correct": False,
                "cost_usd": 0.01,
                "latency_s": 1.0,
                "metadata": "{'sample_id': 's2'}",
            },
            {
                "id": 4,
                "query": "q2",
                "task_type": "gsm8k",
                "selected_model": "strong",
                "selected_budget": 1024,
                "is_correct": True,
                "cost_usd": 0.05,
                "latency_s": 2.0,
                "metadata": "{'sample_id': 's2'}",
            },
        ]
    ).to_csv(csv_path, index=False)

    summary = summarize_baselines(str(csv_path))

    assert "fixed_model_budget_cheap_0" in set(summary["policy"])
    assert "model_only_best_strong" in set(summary["policy"])
    assert "budget_only_best_1024" in set(summary["policy"])
    assert "joint_aggregate_utility" in set(summary["policy_family"])
