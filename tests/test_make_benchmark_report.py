from __future__ import annotations

import pandas as pd

from thinkrouter.experiments import make_benchmark_report


def _write_summary(path, policy: str, accuracy: float, avg_cost: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "policy": policy,
                "accuracy": accuracy,
                "avg_cost": avg_cost,
                "total_cost": avg_cost * 2,
                "avg_latency": 1.0,
                "p95_latency": 1.5,
                "n": 2,
            }
        ]
    ).to_csv(path, index=False)


def test_build_multi_benchmark_report(tmp_path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.csv"
    policy = tmp_path / "policy.csv"
    _write_summary(baseline, "fixed_budget_1024", 0.8, 0.02)
    _write_summary(policy, "fixed_budget_256", 0.8, 0.01)
    monkeypatch.setattr(
        make_benchmark_report,
        "REPORT_INPUTS",
        [
            ("math", "dev20", str(baseline)),
            ("math", "dev20", str(policy)),
        ],
    )

    report = make_benchmark_report.build_report(str(tmp_path / "report.csv"), str(tmp_path / "report.md"))

    assert "cost_vs_fixed_1024" in report.columns
    assert report.loc[report["policy"] == "fixed_budget_256", "cost_vs_fixed_1024"].iloc[0] == 0.5
