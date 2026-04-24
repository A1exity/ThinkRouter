from __future__ import annotations

from thinkrouter.analytics.stability import bootstrap_metric_ci


def test_bootstrap_metric_ci_returns_ordered_interval() -> None:
    summary = bootstrap_metric_ci([0.0, 1.0, 1.0, 0.0, 1.0], num_samples=100, seed=0)
    assert 0.0 <= summary["lower"] <= summary["mean"] <= summary["upper"] <= 1.0
