from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from thinkrouter.experiments.eval_baselines import summarize_baselines
from thinkrouter.experiments.make_failure_taxonomy import summarize_failure_taxonomy
from thinkrouter.experiments.make_final_official_report import build_final_outputs
from thinkrouter.experiments.phase2_router_replay import replay_router
from thinkrouter.experiments.prepare_data import (
    load_gsm8k_samples,
    load_humaneval_samples,
    load_math500_samples,
)
from thinkrouter.experiments.run_grid import run_grid, traces_to_dataframe
from thinkrouter.experiments.datasets import write_samples_jsonl
from thinkrouter.official_protocol import OFFICIAL_PROTOCOL
from thinkrouter.routers import save_factorized_artifact, save_logreg_joint_artifact, train_factorized_router, train_logreg_joint_router
from thinkrouter.training import UtilityObjective, derive_factorized_examples, derive_joint_examples


def prepare_official_data() -> list[Path]:
    written: list[Path] = []
    for benchmark in OFFICIAL_PROTOCOL.benchmarks:
        if benchmark.benchmark == "gsm8k":
            samples = load_gsm8k_samples(benchmark.train_count, benchmark.dev_count, benchmark.test_count)
        elif benchmark.benchmark == "math500":
            samples = load_math500_samples(benchmark.train_count, benchmark.dev_count, benchmark.test_count)
        elif benchmark.benchmark == "humaneval":
            samples = load_humaneval_samples(benchmark.train_count, benchmark.dev_count, benchmark.test_count)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported benchmark: {benchmark.benchmark}")
        out_path = OFFICIAL_PROTOCOL.data_path(benchmark.benchmark)
        write_samples_jsonl(samples, out_path)
        written.append(out_path)
    return written


def run_official_grids(benchmarks: list[str] | None = None, splits: list[str] | None = None) -> list[Path]:
    outputs: list[Path] = []
    benchmark_names = benchmarks or [item.benchmark for item in OFFICIAL_PROTOCOL.benchmarks]
    split_names = splits or ["train", "dev", "test"]
    for benchmark in benchmark_names:
        input_path = OFFICIAL_PROTOCOL.data_path(benchmark)
        for split in split_names:
            traces = run_grid(
                db_path=str(OFFICIAL_PROTOCOL.trace_db_path(benchmark, split)),
                input_path=str(input_path),
                task_type=OFFICIAL_PROTOCOL.benchmark_by_name(benchmark).task_type,
                split=split,
                budgets=list(OFFICIAL_PROTOCOL.budgets),
                model_ids=list(OFFICIAL_PROTOCOL.model_pool),
                resume=True,
            )
            out_path = OFFICIAL_PROTOCOL.grid_csv_path(benchmark, split)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            traces_to_dataframe(traces).to_csv(out_path, index=False)
            outputs.append(out_path)
    return outputs


def _phase2_utility(df: pd.DataFrame) -> pd.DataFrame:
    weights = OFFICIAL_PROTOCOL.utility
    out = df.copy()
    out["utility"] = (
        weights.alpha * pd.to_numeric(out["accuracy"], errors="coerce").fillna(0.0)
        - weights.beta * pd.to_numeric(out["avg_cost"], errors="coerce").fillna(0.0)
        - weights.gamma * pd.to_numeric(out["p95_latency"], errors="coerce").fillna(pd.to_numeric(out.get("avg_latency"), errors="coerce")).fillna(0.0)
    )
    return out


def train_and_replay_official_benchmark(benchmark: str) -> dict[str, str]:
    base = Path("results") / "official" / benchmark
    base.mkdir(parents=True, exist_ok=True)

    train_csv = OFFICIAL_PROTOCOL.grid_csv_path(benchmark, "train")
    dev_csv = OFFICIAL_PROTOCOL.grid_csv_path(benchmark, "dev")
    test_csv = OFFICIAL_PROTOCOL.grid_csv_path(benchmark, "test")
    objective = UtilityObjective(
        alpha=OFFICIAL_PROTOCOL.utility.alpha,
        beta=OFFICIAL_PROTOCOL.utility.beta,
        gamma=OFFICIAL_PROTOCOL.utility.gamma,
    )

    train_df = pd.read_csv(train_csv)
    logreg_artifact = train_logreg_joint_router(derive_joint_examples(train_df, objective))
    factorized_artifact = train_factorized_router(derive_factorized_examples(train_df, objective))

    logreg_path = base / f"{benchmark}_logreg_joint.joblib"
    factorized_path = base / f"{benchmark}_mlp_factorized.joblib"
    save_logreg_joint_artifact(logreg_artifact, str(logreg_path))
    save_factorized_artifact(factorized_artifact, str(factorized_path))

    thresholds = [0.55, 0.65, 0.75, 0.85]
    dev_summaries: list[pd.DataFrame] = []
    for router_name in ["threshold", "logreg_joint", "mlp_factorized"]:
        model_path = None if router_name == "threshold" else str(logreg_path if router_name == "logreg_joint" else factorized_path)
        summary, selected = replay_router(str(dev_csv), router_name, model_path=model_path)
        summary["benchmark"] = benchmark
        dev_summaries.append(summary)
        selected.to_csv(base / f"{benchmark}_dev_{router_name}_selected.csv", index=False)
    best_uncertainty_threshold = thresholds[0]
    best_uncertainty_utility = float("-inf")
    best_uncertainty_summary = pd.DataFrame()
    best_uncertainty_selected = pd.DataFrame()
    for threshold in thresholds:
        summary, selected = replay_router(
            str(dev_csv),
            "uncertainty_aware",
            model_path=str(factorized_path),
            confidence_threshold=threshold,
        )
        score = float(_phase2_utility(summary).iloc[0]["utility"])
        if score > best_uncertainty_utility:
            best_uncertainty_utility = score
            best_uncertainty_threshold = threshold
            best_uncertainty_summary = summary
            best_uncertainty_selected = selected
    best_uncertainty_summary["benchmark"] = benchmark
    dev_summaries.append(best_uncertainty_summary)
    best_uncertainty_selected.to_csv(base / f"{benchmark}_dev_uncertainty_aware_selected.csv", index=False)

    dev_summary = _phase2_utility(pd.concat(dev_summaries, ignore_index=True))
    dev_summary.to_csv(base / f"{benchmark}_dev_phase2_summary.csv", index=False)

    learned_candidates = dev_summary[dev_summary["router_name"].astype(str).isin(["logreg_joint", "mlp_factorized", "uncertainty_aware"])]
    official_learned = learned_candidates.sort_values(["utility", "accuracy", "avg_cost"], ascending=[False, False, True]).iloc[0]
    official_router_name = str(official_learned["router_name"])

    test_rows: list[pd.DataFrame] = []
    selected_by_router: dict[str, pd.DataFrame] = {}
    for router_name in OFFICIAL_PROTOCOL.routers:
        model_path = None
        threshold = None
        if router_name == "logreg_joint":
            model_path = str(logreg_path)
        elif router_name in {"mlp_factorized", "uncertainty_aware"}:
            model_path = str(factorized_path)
        if router_name == "uncertainty_aware":
            threshold = best_uncertainty_threshold
        summary, selected = replay_router(str(test_csv), router_name, model_path=model_path, confidence_threshold=threshold)
        summary["benchmark"] = benchmark
        test_rows.append(summary)
        selected_by_router[router_name] = selected
        selected.to_csv(base / f"{benchmark}_test_{router_name}_selected.csv", index=False)

    baseline_summary = summarize_baselines(str(test_csv))
    combined_summary = pd.concat([baseline_summary, pd.concat(test_rows, ignore_index=True)], ignore_index=True)
    combined_summary = _phase2_utility(combined_summary)
    combined_summary.to_csv(base / f"{benchmark}_test_integrated_summary.csv", index=False)

    official_selected = selected_by_router[official_router_name]
    official_failures = summarize_failure_taxonomy(str(base / f"{benchmark}_test_{official_router_name}_selected.csv"))
    official_failures.to_csv(base / f"{benchmark}_official_learned_failures.csv", index=False)

    selection = pd.DataFrame(
        [
            {
                "benchmark": benchmark,
                "official_router_name": official_router_name,
                "official_router_policy": f"phase2_{official_router_name}",
                "uncertainty_threshold": best_uncertainty_threshold,
                "dev_utility": float(official_learned["utility"]),
            }
        ]
    )
    selection.to_csv(base / f"{benchmark}_router_selection.csv", index=False)
    (base / f"{benchmark}_router_selection.json").write_text(json.dumps(selection.iloc[0].to_dict(), indent=2), encoding="utf-8")
    official_selected.to_csv(base / f"{benchmark}_official_learned_selected.csv", index=False)

    return {
        "benchmark": benchmark,
        "logreg_artifact": str(logreg_path),
        "factorized_artifact": str(factorized_path),
        "dev_summary": str(base / f"{benchmark}_dev_phase2_summary.csv"),
        "test_summary": str(base / f"{benchmark}_test_integrated_summary.csv"),
        "selection": str(base / f"{benchmark}_router_selection.csv"),
    }


def run_final_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_final_outputs()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen official ThinkRouter experiment protocol.")
    parser.add_argument("--stage", default="all", choices=["prepare-data", "grids", "routers", "report", "all"])
    parser.add_argument("--benchmark", action="append", default=None, help="Optional benchmark filter.")
    parser.add_argument("--split", action="append", default=None, help="Optional split filter for --stage grids.")
    args = parser.parse_args()

    if args.stage in {"prepare-data", "all"}:
        for path in prepare_official_data():
            print(f"prepared: {path}")
    if args.stage in {"grids", "all"}:
        for path in run_official_grids(args.benchmark, args.split):
            print(f"grid: {path}")
    if args.stage in {"routers", "all"}:
        for benchmark in args.benchmark or [item.benchmark for item in OFFICIAL_PROTOCOL.benchmarks]:
            outputs = train_and_replay_official_benchmark(benchmark)
            for key, value in outputs.items():
                print(f"{benchmark}:{key}={value}")
    if args.stage in {"report", "all"}:
        results, failures = run_final_report()
        print(results.to_string(index=False))
        if not failures.empty:
            print(failures.to_string(index=False))


if __name__ == "__main__":
    main()
