from __future__ import annotations

import argparse
from pathlib import Path

from thinkrouter.experiments.datasets import summarize_samples, write_samples_jsonl
from thinkrouter.experiments.sample_data import load_frozen_samples


def prepare_seed_data(out_path: str, task_type: str = "all", split: str = "all", limit: int | None = None) -> int:
    samples = load_frozen_samples(task_type=task_type, split=split, limit=limit)
    return write_samples_jsonl(samples, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ThinkRouter benchmark JSONL files.")
    parser.add_argument("--source", default="seed", choices=["seed"], help="Dataset source to export.")
    parser.add_argument("--task", default="all", choices=["gsm8k", "math", "humaneval", "all"], help="Task subset to export.")
    parser.add_argument("--split", default="all", choices=["train", "dev", "test", "all"], help="Split subset to export.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit after filtering.")
    parser.add_argument("--out", default="data/splits/seed.jsonl", help="Output JSONL path.")
    parser.add_argument("--summary", action="store_true", help="Print exported sample counts by task and split.")
    args = parser.parse_args()

    if args.source != "seed":
        raise ValueError("Only --source seed is currently supported.")

    samples = load_frozen_samples(task_type=args.task, split=args.split, limit=args.limit)
    out_path = Path(args.out)
    count = write_samples_jsonl(samples, out_path)
    print(f"Wrote {count} samples to {out_path}")
    if args.summary:
        for (task, split), split_count in sorted(summarize_samples(samples).items()):
            print(f"{task},{split},{split_count}")


if __name__ == "__main__":
    main()