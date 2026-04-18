from __future__ import annotations

import json

import pytest

from thinkrouter.experiments.datasets import load_samples_jsonl, write_samples_jsonl
from thinkrouter.experiments.prepare_data import prepare_seed_data
from thinkrouter.experiments.sample_data import BenchmarkSample


def test_write_and_load_samples_jsonl_filters(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    samples = [
        BenchmarkSample("s1", "gsm8k", "What is 2 + 2?", "4", "train"),
        BenchmarkSample("s2", "math", "Compute 3^2.", "9", "dev"),
    ]

    count = write_samples_jsonl(samples, path)
    loaded = load_samples_jsonl(path, task_type="math", split="dev")

    assert count == 2
    assert loaded == [samples[1]]


def test_load_samples_jsonl_rejects_missing_fields(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"sample_id": "bad"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required fields"):
        load_samples_jsonl(path)


def test_prepare_seed_data_writes_standard_jsonl(tmp_path) -> None:
    path = tmp_path / "seed.jsonl"

    count = prepare_seed_data(str(path), task_type="gsm8k", split="dev", limit=2)
    loaded = load_samples_jsonl(path, task_type="gsm8k", split="dev")

    assert count == 2
    assert len(loaded) == 2
    assert all(sample.sample_id.startswith("gsm8k_") for sample in loaded)

def test_extract_gsm8k_answer_handles_official_format() -> None:
    from thinkrouter.experiments.prepare_data import extract_gsm8k_answer

    assert extract_gsm8k_answer("Natalia sold 48 clips. #### 48") == "48"
    assert extract_gsm8k_answer("The answer is 1,250. #### 1,250") == "1250"
    assert extract_gsm8k_answer("Compute it. #### 12.0") == "12"


def test_extract_math_answer_handles_boxed_solution() -> None:
    from thinkrouter.experiments.prepare_data import extract_math_answer

    assert extract_math_answer(r"We solve and get \boxed{0}.") == "0"
    assert extract_math_answer(r"Thus the answer is \boxed{-\frac{1}{2}}.") == r"-\frac{1}{2}"


def test_load_math_samples_uses_official_fields(monkeypatch) -> None:
    from thinkrouter.experiments import prepare_data

    class FakeSplit:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    def fake_load_dataset(name):
        assert name == "Maxwell-Jia/MATH"
        return {
            "train": FakeSplit([{"problem": "Compute 1+1.", "solution": r"\boxed{2}"}]),
            "test": FakeSplit(
                [
                    {"problem": "Compute 2+1.", "solution": r"\boxed{3}"},
                    {"problem": "Compute 2+2.", "solution": r"\boxed{4}"},
                ]
            ),
        }

    monkeypatch.setattr(prepare_data, "load_dataset", fake_load_dataset, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "datasets", type("FakeDatasets", (), {"load_dataset": staticmethod(fake_load_dataset)}))

    samples = prepare_data.load_math_samples(train_count=1, dev_count=1, test_count=1)

    assert [sample.sample_id for sample in samples] == ["math_train_001", "math_dev_001", "math_test_001"]
    assert [sample.expected_answer for sample in samples] == ["2", "3", "4"]


def test_filter_samples_applies_split_and_limit() -> None:
    from thinkrouter.experiments.prepare_data import filter_samples

    samples = [
        BenchmarkSample("a", "gsm8k", "q", "1", "train"),
        BenchmarkSample("b", "gsm8k", "q", "2", "dev"),
        BenchmarkSample("c", "gsm8k", "q", "3", "dev"),
    ]

    filtered = filter_samples(samples, split="dev", limit=1)

    assert filtered == [samples[1]]
