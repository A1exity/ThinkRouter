from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.run_phase2_eval import run_phase2_eval


def test_run_phase2_eval_writes_expected_outputs(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    pd.DataFrame(
        [
            {
                "id": 1,
                "query": "Add 1 and 2.",
                "task_type": "gsm8k",
                "selected_model": "mock-cheap",
                "selected_model_provider": "mock",
                "selected_model_tier": "cheap",
                "selected_model_alias": "cheap",
                "selected_budget": 0,
                "is_correct": True,
                "cost_usd": 0.001,
                "latency_s": 0.4,
                "metadata": "{'sample_id': 's1'}",
            },
            {
                "id": 2,
                "query": "Add 1 and 2.",
                "task_type": "gsm8k",
                "selected_model": "mock-strong",
                "selected_model_provider": "mock",
                "selected_model_tier": "strong",
                "selected_model_alias": "strong",
                "selected_budget": 1024,
                "is_correct": True,
                "cost_usd": 0.01,
                "latency_s": 1.4,
                "metadata": "{'sample_id': 's1'}",
            },
            {
                "id": 3,
                "query": "Write a Python function factorial(n).",
                "task_type": "humaneval",
                "selected_model": "mock-cheap",
                "selected_model_provider": "mock",
                "selected_model_tier": "cheap",
                "selected_model_alias": "cheap",
                "selected_budget": 0,
                "is_correct": False,
                "cost_usd": 0.001,
                "latency_s": 0.5,
                "metadata": "{'sample_id': 's2'}",
            },
            {
                "id": 4,
                "query": "Write a Python function factorial(n).",
                "task_type": "humaneval",
                "selected_model": "mock-strong",
                "selected_model_provider": "mock",
                "selected_model_tier": "strong",
                "selected_model_alias": "strong",
                "selected_budget": 1024,
                "is_correct": True,
                "cost_usd": 0.01,
                "latency_s": 1.8,
                "metadata": "{'sample_id': 's2'}",
            },
        ]
    ).to_csv(csv_path, index=False)

    outputs = run_phase2_eval(str(csv_path), str(tmp_path / "phase2"))

    assert outputs["logreg_artifact"].endswith("_logreg_joint.joblib")
    assert outputs["factorized_artifact"].endswith("_mlp_factorized.joblib")
    assert outputs["integrated_summary"].endswith("_baseline_phase2_summary.csv")
    for path in outputs.values():
        assert pd.notna(path)
