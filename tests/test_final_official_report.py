from __future__ import annotations

from pathlib import Path

import pandas as pd

from thinkrouter.experiments.make_final_official_report import build_final_outputs


def test_build_final_outputs_from_official_benchmark_bundles(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    benchmark_rows = [
        {
            "policy": "fixed_model_budget_qwen-max_0",
            "policy_family": "fixed_model_budget",
            "accuracy": 0.95,
            "avg_cost": 0.0008,
            "avg_latency": 5.0,
            "p95_latency": 6.0,
        },
        {
            "policy": "aggregate_utility_qwen-plus_256",
            "policy_family": "joint_aggregate_utility",
            "accuracy": 0.9,
            "avg_cost": 0.0005,
            "avg_latency": 4.0,
            "p95_latency": 5.0,
        },
        {
            "policy": "phase2_uncertainty_aware",
            "policy_family": "phase2_uncertainty_aware",
            "router_name": "uncertainty_aware",
            "accuracy": 0.95,
            "avg_cost": 0.0002,
            "avg_latency": 3.0,
            "p95_latency": 4.0,
        },
    ]
    for benchmark in ["gsm8k", "math500", "humaneval"]:
        base = Path("results") / "official" / benchmark
        base.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(benchmark_rows).to_csv(base / f"{benchmark}_test_integrated_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "benchmark": benchmark,
                    "official_router_name": "uncertainty_aware",
                    "official_router_policy": "phase2_uncertainty_aware",
                    "uncertainty_threshold": 0.65,
                    "dev_utility": 0.8,
                }
            ]
        ).to_csv(base / f"{benchmark}_router_selection.csv", index=False)
        pd.DataFrame([{"error_type": "wrong_answer", "count": 2, "avg_cost": 0.0002, "avg_latency": 3.0}]).to_csv(
            base / f"{benchmark}_official_learned_failures.csv",
            index=False,
        )

    results, failures = build_final_outputs()

    assert list(results["benchmark"]) == ["gsm8k", "math500", "humaneval"]
    assert all(results["official_router_name"].astype(str) == "uncertainty_aware")
    assert not failures.empty
    assert Path("results/reports/final_official_report.md").exists()
