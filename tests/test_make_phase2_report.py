from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.make_phase2_report import build_phase2_report


def test_build_phase2_report_ranks_utility_and_writes_outputs(tmp_path) -> None:
    summary_csv = tmp_path / "phase2_summary.csv"
    pd.DataFrame(
        [
            {
                "policy": "aggregate_utility_qwen3-max-2026-01-23_0",
                "policy_family": "joint_aggregate_utility",
                "selected_model": "qwen3-max-2026-01-23",
                "selected_model_tier": "strong",
                "selected_budget": 0,
                "accuracy": 1.0,
                "avg_cost": 0.0008,
                "avg_latency": 4.0,
            },
            {
                "policy": "phase2_mlp_factorized",
                "policy_family": "phase2_mlp_factorized",
                "router_name": "mlp_factorized",
                "selected_model": "qwen3.5-flash-2026-02-23",
                "selected_model_tier": "cheap",
                "selected_budget": "mixed",
                "accuracy": 1.0,
                "avg_cost": 0.0002,
                "avg_latency": 6.0,
                "avg_route_confidence": 0.94,
                "fallback_rate": 0.0,
            },
        ]
    ).to_csv(summary_csv, index=False)

    report = build_phase2_report(str(summary_csv), str(tmp_path / "ranked.csv"), str(tmp_path / "report.md"))

    assert "utility" in report.columns
    assert (tmp_path / "ranked.csv").exists()
    assert (tmp_path / "report.md").exists()
    assert report.loc[report["policy"] == "phase2_mlp_factorized", "utility"].notna().all()
