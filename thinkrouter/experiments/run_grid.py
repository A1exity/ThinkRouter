from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from thinkrouter.app.budgets import budget_to_dict, compile_budget_config
from thinkrouter.app.evaluators import get_evaluator
from thinkrouter.app.models import ModelConfig, build_adapter, default_model_configs, resolve_model_name
from thinkrouter.app.schemas import EvalResult, ModelRequest, ModelResponse, TraceRecord, model_to_dict
from thinkrouter.app.store import TraceStore
from thinkrouter.experiments.datasets import load_samples_jsonl, summarize_samples
from thinkrouter.experiments.sample_data import BenchmarkSample, load_frozen_samples, summarize_frozen_samples
from thinkrouter.runtime import generate_with_runtime

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
        resolved = resolve_model_name(model_id)
        selected.append(
            ModelConfig(
                model_id=resolved.model_id,
                backend=resolved.backend or fallback.backend,
                model_name=resolved.model_name or resolved.model_id,
                base_url=resolved.base_url or fallback.base_url,
                api_key=resolved.api_key or fallback.api_key,
                cost_per_1k_tokens=resolved.cost_per_1k_tokens,
                provider=resolved.provider,
                tier=resolved.tier,
                alias=resolved.alias,
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
    resume: bool = False,
) -> list[TraceRecord]:
    load_dotenv()
    budgets = budgets or DEFAULT_BUDGETS
    experiment = experiment or ("jsonl_grid" if input_path else "frozen_grid")
    samples = load_grid_samples(input_path=input_path, task_type=task_type, split=split, limit=limit)
    configs = select_model_configs(model_ids)
    store = TraceStore(db_path)
    completed = completed_trace_keys(store) if resume else set()
    traces: list[TraceRecord] = []
    for sample in samples:
        traces.extend(_run_sample_grid(sample, configs, budgets, store, experiment, completed))
    return list(store.iter_traces()) if resume else traces


def completed_trace_keys(store: TraceStore) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for trace in store.iter_traces():
        sample_id = str(trace.query_id or trace.metadata.get("sample_id", ""))
        if sample_id:
            keys.add((sample_id, trace.selected_model, int(trace.selected_budget)))
    return keys


def _run_sample_grid(
    sample: BenchmarkSample,
    configs: list[ModelConfig],
    budgets: list[int],
    store: TraceStore,
    experiment: str,
    completed: set[tuple[str, str, int]] | None = None,
) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    evaluator = get_evaluator(sample.task_type)
    completed = completed or set()
    for config in configs:
        adapter = build_adapter(config)
        for budget in budgets:
            if (sample.sample_id, config.model_id, int(budget)) in completed:
                continue
            budget_config = compile_budget_config(int(budget))
            request = ModelRequest(
                query=sample.query,
                task_type=sample.task_type,  # type: ignore[arg-type]
                model_id=config.model_id,
                budget=int(budget),
                budget_config=budget_config,
                candidate_set_signature=_candidate_set_signature(configs, budgets),
                metadata={
                    "expected_answer": sample.expected_answer,
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    **sample.metadata,
                },
            )
            response, runtime_meta = generate_with_runtime(adapter, request)
            if response is None:
                response = ModelResponse(
                    output_text="",
                    parsed_output=None,
                    finish_reason="error",
                    error_type=str(runtime_meta["error_type"]),
                    provider_meta={
                        "provider": config.provider,
                        "tier": config.tier,
                        "budget_id": budget_config.budget_id,
                        "cache_hit": runtime_meta["cache_hit"],
                        "error_message": runtime_meta["error_message"],
                    },
                    raw_response={"error_message": runtime_meta["error_message"], "budget": budget_to_dict(budget_config)},
                )
                evaluation = EvalResult(
                    score=0.0,
                    is_correct=False,
                    parsed_output=None,
                    extracted_answer=None,
                    expected_answer=sample.expected_answer,
                    error_type=str(runtime_meta["error_type"]),
                    judge_metadata={"runtime_error": runtime_meta["error_message"]},
                )
            else:
                evaluation = evaluator.evaluate(response.output_text, sample.expected_answer, request.metadata)
            trace = TraceRecord(
                query_id=sample.sample_id,
                benchmark=sample.task_type,
                query=sample.query,
                query_text=sample.query,
                task_type=sample.task_type,  # type: ignore[arg-type]
                selected_model=config.model_id,
                selected_model_provider=config.provider,
                selected_model_tier=config.tier,
                selected_model_alias=config.alias,
                selected_budget=int(budget),
                selected_budget_id=budget_config.budget_id,
                budget_config=budget_to_dict(budget_config),
                output_text=response.output_text,
                parsed_output=evaluation.parsed_output or response.parsed_output,
                score=evaluation.score,
                is_correct=evaluation.is_correct,
                expected_answer=evaluation.expected_answer,
                extracted_answer=evaluation.extracted_answer,
                judge_metadata=evaluation.judge_metadata,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.total_tokens,
                cost_usd=adapter.estimate_cost(response.total_tokens),
                latency_s=response.latency_s,
                error_type=evaluation.error_type or response.error_type,
                candidate_set_signature=request.candidate_set_signature,
                prompt_template_version=budget_config.prompt_template_version,
                provider_response_meta=response.provider_meta,
                metadata={
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "experiment": experiment,
                    "cache_hit": runtime_meta["cache_hit"],
                    "selected_model_provider": config.provider,
                    "selected_model_tier": config.tier,
                    "selected_model_alias": config.alias,
                },
            )
            traces.append(store.insert_trace(trace))
    return traces


def traces_to_dataframe(traces: list[TraceRecord]) -> pd.DataFrame:
    return pd.DataFrame([model_to_dict(trace) for trace in traces])


def _candidate_set_signature(configs: list[ModelConfig], budgets: list[int]) -> str:
    model_ids = ",".join(config.model_id for config in configs)
    budget_ids = ",".join(str(int(budget)) for budget in budgets)
    return f"models={model_ids}|budgets={budget_ids}"


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
    parser.add_argument("--resume", action="store_true", help="Skip sample/model/budget traces already present in --db and export all rows from that DB.")
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
        resume=args.resume,
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
