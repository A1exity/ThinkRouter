from __future__ import annotations

import pandas as pd

from thinkrouter.experiments import make_gsm8k_report


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


def test_build_report_writes_summary_markdown_and_figure(tmp_path, monkeypatch) -> None:
    fixed = tmp_path / "fixed.csv"
    learned = tmp_path / "learned.csv"
    _write_summary(fixed, "fixed_budget_1024", 0.9, 0.02)
    _write_summary(learned, "safe_learned_policy_fallback_budget_256", 0.9, 0.01)
    monkeypatch.setattr(
        make_gsm8k_report,
        "REPORT_INPUTS",
        [
            ("test20", "fixed_or_oracle", str(fixed)),
            ("test20", "dev_calibrated", str(learned)),
        ],
    )

    summary_out = tmp_path / "report.csv"
    markdown_out = tmp_path / "report.md"
    figure_out = tmp_path / "report.png"
    report = make_gsm8k_report.build_report(str(summary_out), str(markdown_out), str(figure_out))

    assert summary_out.exists()
    assert markdown_out.exists()
    assert figure_out.exists()
    assert "policy_family" in report.columns
    assert "cost_vs_fixed_1024" in report.columns
    assert report.loc[report["policy"] == "safe_learned_policy_fallback_budget_256", "cost_vs_fixed_1024"].iloc[0] == 0.5
