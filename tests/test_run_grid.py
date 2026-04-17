from __future__ import annotations

from thinkrouter.experiments.run_grid import parse_csv_ints, run_grid
from thinkrouter.experiments.sample_data import load_frozen_samples, summarize_frozen_samples


def test_load_frozen_samples_filters_task_and_split() -> None:
    dev_math = load_frozen_samples(task_type="math", split="dev")

    assert len(dev_math) == 2
    assert all(sample.task_type == "math" for sample in dev_math)
    assert all(sample.split == "dev" for sample in dev_math)


def test_frozen_sample_summary_contains_seed_splits() -> None:
    summary = summarize_frozen_samples()

    assert summary[("gsm8k", "train")] == 12
    assert summary[("math", "dev")] == 2
    assert summary[("humaneval", "test")] == 2


def test_run_grid_runs_task_split_mock_grid(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")

    traces = run_grid(
        db_path=str(tmp_path / "grid.sqlite"),
        task_type="math",
        split="dev",
        budgets=[0, 256],
    )

    assert len(traces) == 8
    assert {trace.task_type for trace in traces} == {"math"}
    assert {trace.selected_budget for trace in traces} == {0, 256}
    assert all(trace.metadata["split"] == "dev" for trace in traces)


def test_run_grid_limit_applies_to_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")

    traces = run_grid(
        db_path=str(tmp_path / "grid.sqlite"),
        task_type="all",
        split="train",
        budgets=parse_csv_ints("0,1024"),
        limit=1,
    )

    assert len(traces) == 4
    assert len({trace.metadata["sample_id"] for trace in traces}) == 1

def test_run_grid_uses_jsonl_input(tmp_path, monkeypatch) -> None:
    from thinkrouter.experiments.datasets import write_samples_jsonl
    from thinkrouter.experiments.sample_data import BenchmarkSample

    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")
    input_path = tmp_path / "samples.jsonl"
    write_samples_jsonl(
        [
            BenchmarkSample("jsonl_1", "gsm8k", "What is 6 + 1?", "7", "dev"),
            BenchmarkSample("jsonl_2", "math", "Compute 2^3.", "8", "train"),
        ],
        input_path,
    )

    traces = run_grid(
        db_path=str(tmp_path / "grid.sqlite"),
        input_path=str(input_path),
        task_type="gsm8k",
        split="dev",
        budgets=[0],
        model_ids=["mock-cheap"],
    )

    assert len(traces) == 1
    assert traces[0].metadata["sample_id"] == "jsonl_1"
    assert traces[0].metadata["experiment"] == "jsonl_grid"