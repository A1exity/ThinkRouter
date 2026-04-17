from __future__ import annotations

from thinkrouter.experiments.run_day1_grid import run_day1_grid


def test_day1_grid_runs_small_mock_grid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")
    traces = run_day1_grid(str(tmp_path / "day1.sqlite"), limit=2, budgets=[0, 256])
    assert len(traces) == 8
    assert all(trace.task_type == "gsm8k" for trace in traces)
