from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from thinkrouter.experiments.datasets import summarize_samples, write_samples_jsonl
from thinkrouter.experiments.sample_data import BenchmarkSample, load_frozen_samples

DEFAULT_GSM8K_COUNTS = {"train": 60, "dev": 20, "test": 20}
DEFAULT_MATH_COUNTS = {"train": 60, "dev": 20, "test": 20}
DEFAULT_HUMANEVAL_COUNTS = {"train": 60, "dev": 20, "test": 20}
MATH_DATASET_ID = "Maxwell-Jia/MATH"
MATH500_DATASET_ID = "HuggingFaceH4/MATH-500"
HUMANEVAL_DATASET_IDS = ("openai/openai_humaneval", "openai_humaneval")


def prepare_seed_data(out_path: str, task_type: str = "all", split: str = "all", limit: int | None = None) -> int:
    samples = load_frozen_samples(task_type=task_type, split=split, limit=limit)
    return write_samples_jsonl(samples, out_path)


def extract_gsm8k_answer(answer: str) -> str:
    if "####" in answer:
        final = answer.split("####")[-1]
    else:
        final = answer
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", final)
    if not numbers:
        raise ValueError(f"Could not extract numeric answer from GSM8K answer: {answer[:120]!r}")
    raw = numbers[-1].replace(",", "")
    if "." in raw:
        raw = raw.rstrip("0").rstrip(".") or "0"
    return raw


def extract_math_answer(solution: str) -> str:
    boxed = _extract_last_boxed(solution)
    if boxed:
        return boxed
    marker_match = re.search(r"(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)", solution, flags=re.IGNORECASE)
    if marker_match:
        return marker_match.group(1).strip()
    raise ValueError(f"Could not extract boxed MATH answer from solution: {solution[:120]!r}")


def _extract_last_boxed(text: str) -> str | None:
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        chars: list[str] = []
        index = start
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = "".join(chars).strip()
                    return value or None
                chars.append(char)
            else:
                chars.append(char)
            index += 1
    return None


def _take_rows(dataset: Any, count: int) -> list[dict[str, Any]]:
    return [dataset[index] for index in range(min(count, len(dataset)))]


def load_gsm8k_samples(train_count: int = 60, dev_count: int = 20, test_count: int = 20) -> list[BenchmarkSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for --source gsm8k. It is listed in environment.yml.") from exc

    dataset = load_dataset("openai/gsm8k", "main")
    train_rows = _take_rows(dataset["train"], train_count)
    test_rows = _take_rows(dataset["test"], dev_count + test_count)
    samples: list[BenchmarkSample] = []
    for index, row in enumerate(train_rows, start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"gsm8k_train_{index:03d}",
                task_type="gsm8k",
                query=str(row["question"]),
                expected_answer=extract_gsm8k_answer(str(row["answer"])),
                split="train",
            )
        )
    for index, row in enumerate(test_rows[:dev_count], start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"gsm8k_dev_{index:03d}",
                task_type="gsm8k",
                query=str(row["question"]),
                expected_answer=extract_gsm8k_answer(str(row["answer"])),
                split="dev",
            )
        )
    for index, row in enumerate(test_rows[dev_count : dev_count + test_count], start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"gsm8k_test_{index:03d}",
                task_type="gsm8k",
                query=str(row["question"]),
                expected_answer=extract_gsm8k_answer(str(row["answer"])),
                split="test",
            )
        )
    return samples


def load_math_samples(train_count: int = 60, dev_count: int = 20, test_count: int = 20) -> list[BenchmarkSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for --source math. It is listed in environment.yml.") from exc

    dataset = load_dataset(MATH_DATASET_ID)
    train_rows = _take_rows(dataset["train"], train_count)
    test_rows = _take_rows(dataset["test"], dev_count + test_count)
    samples: list[BenchmarkSample] = []
    for index, row in enumerate(train_rows, start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"math_train_{index:03d}",
                task_type="math",
                query=str(row["problem"]),
                expected_answer=extract_math_answer(str(row["solution"])),
                split="train",
            )
        )
    for index, row in enumerate(test_rows[:dev_count], start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"math_dev_{index:03d}",
                task_type="math",
                query=str(row["problem"]),
                expected_answer=extract_math_answer(str(row["solution"])),
                split="dev",
            )
        )
    for index, row in enumerate(test_rows[dev_count : dev_count + test_count], start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"math_test_{index:03d}",
                task_type="math",
                query=str(row["problem"]),
                expected_answer=extract_math_answer(str(row["solution"])),
                split="test",
            )
        )
    return samples


def load_math500_samples(train_count: int = 60, dev_count: int = 20, test_count: int = 20) -> list[BenchmarkSample]:
    dataset = _load_first_available_dataset([os.getenv("THINKROUTER_MATH500_DATASET_ID", MATH500_DATASET_ID)])
    rows = _take_rows(dataset["test"], train_count + dev_count + test_count)
    samples: list[BenchmarkSample] = []
    for index, row in enumerate(rows[:train_count], start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"math500_train_{index:03d}",
                task_type="math",
                query=str(row["problem"]),
                expected_answer=extract_math_answer(str(row["solution"])),
                split="train",
                metadata={"benchmark": "math500"},
            )
        )
    for index, row in enumerate(rows[train_count : train_count + dev_count], start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"math500_dev_{index:03d}",
                task_type="math",
                query=str(row["problem"]),
                expected_answer=extract_math_answer(str(row["solution"])),
                split="dev",
                metadata={"benchmark": "math500"},
            )
        )
    for index, row in enumerate(rows[train_count + dev_count : train_count + dev_count + test_count], start=1):
        samples.append(
            BenchmarkSample(
                sample_id=f"math500_test_{index:03d}",
                task_type="math",
                query=str(row["problem"]),
                expected_answer=extract_math_answer(str(row["solution"])),
                split="test",
                metadata={"benchmark": "math500"},
            )
        )
    return samples


def load_humaneval_samples(train_count: int = 60, dev_count: int = 20, test_count: int = 20) -> list[BenchmarkSample]:
    dataset = _load_first_available_dataset(list(HUMANEVAL_DATASET_IDS))
    split_name = "test" if "test" in dataset else next(iter(dataset.keys()))
    rows = _take_rows(dataset[split_name], train_count + dev_count + test_count)
    samples: list[BenchmarkSample] = []
    for sample_split, offset, count in [("train", 0, train_count), ("dev", train_count, dev_count), ("test", train_count + dev_count, test_count)]:
        for index, row in enumerate(rows[offset : offset + count], start=1):
            entry_point = str(row.get("entry_point") or "")
            canonical_solution = str(row.get("canonical_solution") or row.get("canonical_solution_text") or "")
            test_code = str(row.get("test") or row.get("test_code") or "")
            samples.append(
                BenchmarkSample(
                    sample_id=f"humaneval_{sample_split}_{index:03d}",
                    task_type="humaneval",
                    query=str(row["prompt"]),
                    expected_answer="pass",
                    split=sample_split,
                    metadata={
                        "benchmark": "humaneval",
                        "entry_point": entry_point,
                        "canonical_solution": canonical_solution,
                        "test_code": test_code,
                    },
                )
            )
    return samples


def _load_first_available_dataset(dataset_ids: list[str]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for official benchmark export. It is listed in environment.yml.") from exc

    last_error: Exception | None = None
    for dataset_id in dataset_ids:
        try:
            return load_dataset(dataset_id)
        except Exception as exc:  # pragma: no cover - network / hub errors vary by environment
            last_error = exc
            continue
    raise RuntimeError(f"Could not load any dataset from {dataset_ids!r}") from last_error


def filter_samples(samples: list[BenchmarkSample], split: str = "all", limit: int | None = None) -> list[BenchmarkSample]:
    filtered = samples
    if split != "all":
        filtered = [sample for sample in filtered if sample.split == split]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ThinkRouter benchmark JSONL files.")
    parser.add_argument("--source", default="seed", choices=["seed", "gsm8k", "math", "math500", "humaneval"], help="Dataset source to export.")
    parser.add_argument("--task", default="all", choices=["gsm8k", "math", "humaneval", "all"], help="Task subset to export. Only used by --source seed.")
    parser.add_argument("--split", default="all", choices=["train", "dev", "test", "all"], help="Split subset to export.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit after filtering.")
    parser.add_argument("--out", default="data/splits/seed.jsonl", help="Output JSONL path.")
    parser.add_argument("--gsm8k-train", type=int, default=DEFAULT_GSM8K_COUNTS["train"], help="Number of GSM8K train examples to export.")
    parser.add_argument("--gsm8k-dev", type=int, default=DEFAULT_GSM8K_COUNTS["dev"], help="Number of GSM8K dev examples to export from the official test split.")
    parser.add_argument("--gsm8k-test", type=int, default=DEFAULT_GSM8K_COUNTS["test"], help="Number of GSM8K test examples to export from the official test split after dev examples.")
    parser.add_argument("--math-train", type=int, default=DEFAULT_MATH_COUNTS["train"], help="Number of MATH train examples to export.")
    parser.add_argument("--math-dev", type=int, default=DEFAULT_MATH_COUNTS["dev"], help="Number of MATH dev examples to export from the official test split.")
    parser.add_argument("--math-test", type=int, default=DEFAULT_MATH_COUNTS["test"], help="Number of MATH test examples to export from the official test split after dev examples.")
    parser.add_argument("--humaneval-train", type=int, default=DEFAULT_HUMANEVAL_COUNTS["train"], help="Number of HumanEval train examples to export from the official pool.")
    parser.add_argument("--humaneval-dev", type=int, default=DEFAULT_HUMANEVAL_COUNTS["dev"], help="Number of HumanEval dev examples to export from the official pool.")
    parser.add_argument("--humaneval-test", type=int, default=DEFAULT_HUMANEVAL_COUNTS["test"], help="Number of HumanEval test examples to export from the official pool.")
    parser.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint mirror, for example https://hf-mirror.com.")
    parser.add_argument("--summary", action="store_true", help="Print exported sample counts by task and split.")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    if args.source == "seed":
        samples = load_frozen_samples(task_type=args.task, split=args.split, limit=args.limit)
    elif args.source == "gsm8k":
        samples = load_gsm8k_samples(train_count=args.gsm8k_train, dev_count=args.gsm8k_dev, test_count=args.gsm8k_test)
        samples = filter_samples(samples, split=args.split, limit=args.limit)
    elif args.source == "math":
        samples = load_math_samples(train_count=args.math_train, dev_count=args.math_dev, test_count=args.math_test)
        samples = filter_samples(samples, split=args.split, limit=args.limit)
    elif args.source == "math500":
        samples = load_math500_samples(train_count=args.math_train, dev_count=args.math_dev, test_count=args.math_test)
        samples = filter_samples(samples, split=args.split, limit=args.limit)
    elif args.source == "humaneval":
        samples = load_humaneval_samples(train_count=args.humaneval_train, dev_count=args.humaneval_dev, test_count=args.humaneval_test)
        samples = filter_samples(samples, split=args.split, limit=args.limit)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    out_path = Path(args.out)
    count = write_samples_jsonl(samples, out_path)
    print(f"Wrote {count} samples to {out_path}")
    if args.summary:
        for (task, split), split_count in sorted(summarize_samples(samples).items()):
            print(f"{task},{split},{split_count}")


if __name__ == "__main__":
    main()
