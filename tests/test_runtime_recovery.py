from __future__ import annotations

from thinkrouter.experiments.run_grid import run_grid


class _ExplodingAdapter:
    def estimate_cost(self, total_tokens: int) -> float:
        return 0.0

    def generate(self, request):
        raise TimeoutError("provider timed out")


def test_run_grid_records_failed_trace_instead_of_crashing(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_MODEL_POOL", "")
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_MID_MODEL", "")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "")
    monkeypatch.setenv("THINKROUTER_DISABLE_CACHE", "1")
    monkeypatch.setattr("thinkrouter.experiments.run_grid.build_adapter", lambda config: _ExplodingAdapter())

    traces = run_grid(
        db_path=str(tmp_path / "grid.sqlite"),
        task_type="gsm8k",
        split="dev",
        budgets=[0],
        model_ids=["mock-cheap"],
        limit=1,
    )

    assert len(traces) == 1
    assert traces[0].is_correct is False
    assert traces[0].error_type == "TimeoutError"
