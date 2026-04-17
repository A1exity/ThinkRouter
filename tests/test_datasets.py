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


def test_filter_samples_applies_split_and_limit() -> None:
    from thinkrouter.experiments.prepare_data import filter_samples

    samples = [
        BenchmarkSample("a", "gsm8k", "q", "1", "train"),
        BenchmarkSample("b", "gsm8k", "q", "2", "dev"),
        BenchmarkSample("c", "gsm8k", "q", "3", "dev"),
    ]

    filtered = filter_samples(samples, split="dev", limit=1)

    assert filtered == [samples[1]]