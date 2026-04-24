from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.make_ablation_report import build_ablation_report


def test_build_ablation_report_writes_outputs(tmp_path) -> None:
    summary_csv = tmp_path / "summary.csv"
    pd.DataFrame(
        [
            {"policy": "phase2_threshold", "policy_family": "phase2_threshold", "accuracy": 1.0, "avg_cost": 0.1, "avg_latency": 1.0, "utility": 0.8},
            {"policy": "phase2_mlp_factorized", "policy_family": "phase2_mlp_factorized", "accuracy": 1.0, "avg_cost": 0.2, "avg_latency": 2.0, "utility": 0.7},
        ]
    ).to_csv(summary_csv, index=False)

    report = build_ablation_report([str(summary_csv)], str(tmp_path / "ablation.csv"), str(tmp_path / "ablation.md"))

    assert "ablation_group" in report.columns
    assert (tmp_path / "ablation.csv").exists()
    assert (tmp_path / "ablation.md").exists()
