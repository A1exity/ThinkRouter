from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from thinkrouter.experiments.sample_data import BenchmarkSample, VALID_SPLITS, VALID_TASKS

REQUIRED_JSONL_FIELDS = {"sample_id", "task_type", "query", "expected_answer", "split"}


def sample_to_json(sample: BenchmarkSample) -> dict[str, str]:
    return {
        "sample_id": sample.sample_id,
        "task_type": sample.task_type,
        "query": sample.query,
        "expected_answer": sample.expected_answer,
        "split": sample.split,
    }


def sample_from_json(row: dict[str, Any], line_number: int | None = None) -> BenchmarkSample:
    missing = REQUIRED_JSONL_FIELDS - set(row)
    if missing:
        location = f" on line {line_number}" if line_number is not None else ""
        raise ValueError(f"Missing required fields{location}: {sorted(missing)}")
    sample = BenchmarkSample(
        sample_id=str(row["sample_id"]),
        task_type=str(row["task_type"]),
        query=str(row["query"]),
        expected_answer=str(row["expected_answer"]),
        split=str(row["split"]),
    )
    validate_sample(sample, line_number=line_number)
    return sample


def validate_sample(sample: BenchmarkSample, line_number: int | None = None) -> None:
    location = f" on line {line_number}" if line_number is not None else ""
    if not sample.sample_id.strip():
        raise ValueError(f"Empty sample_id{location}")
    if sample.task_type not in VALID_TASKS - {"all"}:
        raise ValueError(f"Unsupported task_type{location}: {sample.task_type}")
    if sample.split not in VALID_SPLITS - {"all"}:
        raise ValueError(f"Unsupported split{location}: {sample.split}")
    if not sample.query.strip():
        raise ValueError(f"Empty query{location}")
    if not sample.expected_answer.strip():
        raise ValueError(f"Empty expected_answer{location}")


def write_samples_jsonl(samples: Iterable[BenchmarkSample], path: str | Path) -> int:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            validate_sample(sample)
            handle.write(json.dumps(sample_to_json(sample), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_samples_jsonl(
    path: str | Path,
    task_type: str = "all",
    split: str = "all",
    limit: int | None = None,
) -> list[BenchmarkSample]:
    if task_type not in VALID_TASKS:
        raise ValueError(f"Unsupported task_type: {task_type}. Expected one of {sorted(VALID_TASKS)}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}. Expected one of {sorted(VALID_SPLITS)}")
    samples: list[BenchmarkSample] = []
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc.msg}") from exc
            sample = sample_from_json(row, line_number=line_number)
            if task_type != "all" and sample.task_type != task_type:
                continue
            if split != "all" and sample.split != split:
                continue
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    return samples


def summarize_samples(samples: Iterable[BenchmarkSample]) -> dict[tuple[str, str], int]:
    summary: dict[tuple[str, str], int] = {}
    for sample in samples:
        key = (sample.task_type, sample.split)
        summary[key] = summary.get(key, 0) + 1
    return summary