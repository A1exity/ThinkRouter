from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.run_eval import main


def test_run_eval_writes_summary_ranked_and_plot(tmp_path, monkeypatch) -> None:
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
                "latency_s": 0.5,
                "total_tokens": 10,
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
                "latency_s": 1.5,
                "total_tokens": 30,
                "metadata": "{'sample_id': 's1'}",
            },
        ]
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr("sys.argv", ["run_eval", str(csv_path), "--out-prefix", str(tmp_path / "eval")])
    main()

    assert (tmp_path / "eval_baseline_summary.csv").exists()
    assert (tmp_path / "eval_ranked.csv").exists()
    assert (tmp_path / "eval_report.md").exists()
    assert (tmp_path / "eval_pareto.png").exists()
