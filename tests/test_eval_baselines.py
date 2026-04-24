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
    assert "joint_safe_fallback" in set(summary["policy_family"])


def test_summarize_baselines_can_append_phase2_router(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "query": "Add 1 and 2.",
                "task_type": "gsm8k",
                "selected_model": "mock-cheap",
                "selected_budget": 0,
                "is_correct": True,
                "cost_usd": 0.01,
                "latency_s": 1.0,
                "selected_model_provider": "mock",
                "selected_model_tier": "cheap",
                "selected_model_alias": "cheap",
                "metadata": "{'sample_id': 's1'}",
            },
            {
                "id": 2,
                "query": "Write a Python function factorial(n).",
                "task_type": "humaneval",
                "selected_model": "mock-strong",
                "selected_budget": 1024,
                "is_correct": True,
                "cost_usd": 0.05,
                "latency_s": 2.0,
                "selected_model_provider": "mock",
                "selected_model_tier": "strong",
                "selected_model_alias": "strong",
                "metadata": "{'sample_id': 's2'}",
            },
        ]
    ).to_csv(csv_path, index=False)

    summary = summarize_baselines(str(csv_path), phase2_routers=["threshold"])

    phase2 = summary.loc[summary["policy"] == "phase2_threshold"]
    assert not phase2.empty
    assert phase2["policy_family"].iloc[0] == "phase2_threshold"
