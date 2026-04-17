from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from thinkrouter.app.evaluators import get_evaluator
from thinkrouter.app.models import ModelConfig, build_adapter, default_model_configs
from thinkrouter.app.schemas import ModelRequest, TraceRecord, model_to_dict
from thinkrouter.app.store import TraceStore
from thinkrouter.experiments.datasets import load_samples_jsonl, summarize_samples
from thinkrouter.experiments.sample_data import BenchmarkSample, load_frozen_samples, summarize_frozen_samples

DEFAULT_BUDGETS = [0, 256, 1024]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def select_model_configs(model_ids: list[str] | None = None) -> list[ModelConfig]:
    configs = default_model_configs()
    if model_ids is None:
        return list(configs.values())
    selected: list[ModelConfig] = []
    fallback = next(iter(configs.values()))
    for model_id in model_ids:
        if model_id in configs:
            selected.append(configs[model_id])
            continue
        backend = "mock" if model_id.startswith("mock") else fallback.backend
        selected.append(
            ModelConfig(
                model_id=model_id,
                backend=backend,
                model_name=model_id,
                base_url=fallback.base_url,
                api_key=fallback.api_key,
                cost_per_1k_tokens=fallback.cost_per_1k_tokens,
            )
        )
    return selected


def load_grid_samples(
    input_path: str | None = None,
    task_type: str = "all",
    split: str = "dev",
    limit: int | None = None,
) -> list[BenchmarkSample]:
    if input_path:
        return load_samples_jsonl(input_path, task_type=task_type, split=split, limit=limit)
    return load_frozen_samples(task_type=task_type, split=split, limit=limit)


def run_grid(
    db_path: str,
    task_type: str = "all",
    split: str = "dev",
    budgets: list[int] | None = None,
    model_ids: list[str] | None = None,
    limit: int | None = None,
    experiment: str | None = None,
    input_path: str | None = None,
) -> list[TraceRecord]:
    load_dotenv()
    budgets = budgets or DEFAULT_BUDGETS
    experiment = experiment or ("jsonl_grid" if input_path else "frozen_grid")
    samples = load_grid_samples(input_path=input_path, task_type=task_type, split=split, limit=limit)
    configs = select_model_configs(model_ids)
    store = TraceStore(db_path)
    traces: list[TraceRecord] = []
    for sample in samples:
        traces.extend(_run_sample_grid(sample, configs, budgets, store, experiment))
    return traces


def _run_sample_grid(
    sample: BenchmarkSample,
    configs: list[ModelConfig],
    budgets: list[int],
    store: TraceStore,
    experiment: str,
) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    evaluator = get_evaluator(sample.task_type)
    for config in configs:
        adapter = build_adapter(config)
        for budget in budgets:
            request = ModelRequest(
                query=sample.query,
                task_type=sample.task_type,  # type: ignore[arg-type]
                model_id=config.model_id,
                budget=budget,
                metadata={"expected_answer": sample.expected_answer, "sample_id": sample.sample_id, "split": sample.split},
            )
            response = adapter.generate(request)
            evaluation = evaluator.evaluate(response.output_text, sample.expected_answer)
            trace = TraceRecord(
                query=sample.query,
                task_type=sample.task_type,  # type: ignore[arg-type]
                selected_model=config.model_id,
                selected_budget=budget,
                output_text=response.output_text,
                score=evaluation.score,
                is_correct=evaluation.is_correct,
                expected_answer=evaluation.expected_answer,
                extracted_answer=evaluation.extracted_answer,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                cost_usd=adapter.estimate_cost(response.total_tokens),
                latency_s=response.latency_s,
                metadata={"sample_id": sample.sample_id, "split": sample.split, "experiment": experiment},
            )
            traces.append(store.insert_trace(trace))
    return traces


def traces_to_dataframe(traces: list[TraceRecord]) -> pd.DataFrame:
    return pd.DataFrame([model_to_dict(trace) for trace in traces])


def _print_summary(input_path: str | None) -> None:
    if input_path:
        samples = load_samples_jsonl(input_path, task_type="all", split="all")
        summary = summarize_samples(samples)
    else:
        summary = summarize_frozen_samples()
    for (task, split), count in sorted(summary.items()):
        print(f"{task},{split},{count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a ThinkRouter grid over frozen seed splits or a benchmark JSONL file.")
    parser.add_argument("--input", default=None, help="Optional benchmark JSONL path. Defaults to built-in frozen seed samples.")
    parser.add_argument("--db", default="results/traces/grid.sqlite", help="SQLite trace database path.")
    parser.add_argument("--out", default="results/tables/grid.csv", help="CSV output path.")
    parser.add_argument("--task", default="all", choices=["gsm8k", "math", "humaneval", "all"], help="Task subset to run.")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "all"], help="Split to run.")
    parser.add_argument("--budgets", default="0,256,1024", help="Comma-separated budget levels.")
    parser.add_argument("--models", default=None, help="Comma-separated model ids. Defaults to configured cheap and strong models.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit after task/split filtering.")
    parser.add_argument("--summary", action="store_true", help="Print sample counts and exit.")
    args = parser.parse_args()

    if args.summary:
        _print_summary(args.input)
        return

    traces = run_grid(
        db_path=args.db,
        task_type=args.task,
        split=args.split,
        budgets=parse_csv_ints(args.budgets),
        model_ids=parse_csv_strings(args.models),
        limit=args.limit,
        input_path=args.input,
    )
    df = traces_to_dataframe(traces)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    accuracy = df["is_correct"].mean() if not df.empty else 0.0
    print(f"Wrote {len(df)} traces to {args.db}")
    print(f"Wrote table to {out_path}")
    print(f"Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()