from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from thinkrouter.app.budgets import budget_to_dict, compile_budget_config
from thinkrouter.app.evaluators import get_evaluator
from thinkrouter.app.models import build_adapter, default_model_configs
from thinkrouter.app.schemas import ModelRequest, TraceRecord, model_to_dict
from thinkrouter.app.store import TraceStore
from thinkrouter.experiments.sample_data import load_day1_samples


DEFAULT_BUDGETS = [0, 256, 1024]


def run_day1_grid(db_path: str, limit: int = 20, budgets: list[int] | None = None) -> list[TraceRecord]:
    load_dotenv()
    budgets = budgets or DEFAULT_BUDGETS
    configs = default_model_configs()
    selected_configs = list(configs.values())[:3]
    if len(selected_configs) == 1:
        selected_configs.append(selected_configs[0])
    store = TraceStore(db_path)
    traces: list[TraceRecord] = []
    for sample in load_day1_samples(limit):
        for config in selected_configs:
            adapter = build_adapter(config)
            for budget in budgets:
                budget_config = compile_budget_config(int(budget))
                request = ModelRequest(
                    query=sample.query,
                    task_type="gsm8k",
                    model_id=config.model_id,
                    budget=int(budget),
                    budget_config=budget_config,
                    candidate_set_signature=f"models={','.join(item.model_id for item in selected_configs)}|budgets={','.join(str(int(item)) for item in budgets)}",
                    metadata={"expected_answer": sample.expected_answer, "sample_id": sample.sample_id, **sample.metadata},
                )
                response = adapter.generate(request)
                evaluation = get_evaluator("gsm8k").evaluate(response.output_text, sample.expected_answer, request.metadata)
                trace = TraceRecord(
                    query_id=sample.sample_id,
                    benchmark="gsm8k",
                    query=sample.query,
                    query_text=sample.query,
                    task_type="gsm8k",
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
                        "experiment": "day1_grid",
                        "selected_model_provider": config.provider,
                        "selected_model_tier": config.tier,
                        "selected_model_alias": config.alias,
                    },
                )
                traces.append(store.insert_trace(trace))
    return traces


def traces_to_dataframe(traces: list[TraceRecord]) -> pd.DataFrame:
    return pd.DataFrame([model_to_dict(trace) for trace in traces])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day-1 ThinkRouter GSM8K grid.")
    parser.add_argument("--db", default="results/traces/day1.sqlite", help="SQLite trace database path.")
    parser.add_argument("--limit", type=int, default=20, help="Number of built-in GSM8K samples to run.")
    parser.add_argument("--out", default="results/tables/day1_grid.csv", help="CSV output path.")
    args = parser.parse_args()

    traces = run_day1_grid(args.db, limit=args.limit)
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

