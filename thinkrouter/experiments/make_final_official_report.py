from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thinkrouter.official_protocol import OFFICIAL_PROTOCOL


def add_utility(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    weights = OFFICIAL_PROTOCOL.utility
    out["utility"] = (
        weights.alpha * pd.to_numeric(out["accuracy"], errors="coerce").fillna(0.0)
        - weights.beta * pd.to_numeric(out["avg_cost"], errors="coerce").fillna(0.0)
        - weights.gamma * pd.to_numeric(out["p95_latency"], errors="coerce").fillna(pd.to_numeric(out.get("avg_latency"), errors="coerce")).fillna(0.0)
    )
    return out


def load_benchmark_bundle(benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path("results") / "official" / benchmark
    summary = pd.read_csv(base / f"{benchmark}_test_integrated_summary.csv")
    selection = pd.read_csv(base / f"{benchmark}_router_selection.csv")
    failures = pd.read_csv(base / f"{benchmark}_official_learned_failures.csv") if (base / f"{benchmark}_official_learned_failures.csv").exists() else pd.DataFrame()
    return add_utility(summary), selection, failures


def select_final_rows(summary: pd.DataFrame, selection: pd.DataFrame, benchmark: str) -> dict[str, object]:
    fixed_rows = summary[summary["policy_family"] == "fixed_model_budget"].copy()
    strongest_fixed = fixed_rows.sort_values(["utility", "accuracy", "avg_cost"], ascending=[False, False, True]).iloc[0]
    aggregate = summary[summary["policy_family"] == "joint_aggregate_utility"].sort_values(["utility"], ascending=False).iloc[0]
    learned_policy = str(selection.iloc[0]["official_router_policy"])
    learned = summary[summary["policy"].astype(str) == learned_policy].iloc[0]
    return {
        "benchmark": benchmark,
        "official_learned_policy": learned["policy"],
        "official_router_name": learned.get("router_name", pd.NA),
        "accuracy": float(learned["accuracy"]),
        "avg_cost": float(learned["avg_cost"]),
        "p95_latency": float(learned["p95_latency"]),
        "utility": float(learned["utility"]),
        "strongest_fixed_policy": strongest_fixed["policy"],
        "strongest_fixed_utility": float(strongest_fixed["utility"]),
        "aggregate_policy": aggregate["policy"],
        "aggregate_utility": float(aggregate["utility"]),
        "cost_reduction_vs_strongest_fixed": float((float(strongest_fixed["avg_cost"]) - float(learned["avg_cost"])) / max(float(strongest_fixed["avg_cost"]), 1e-9)),
        "accuracy_drop_vs_strongest_fixed": float(float(strongest_fixed["accuracy"]) - float(learned["accuracy"])),
        "beats_strongest_fixed": float(learned["utility"]) > float(strongest_fixed["utility"]),
        "beats_aggregate_baseline": float(learned["utility"]) > float(aggregate["utility"]),
    }


def build_final_outputs(
    summary_out: str = "results/tables/final_official_results.csv",
    figure_out: str = "results/figures/final_official_pareto.png",
    failures_out: str = "results/tables/final_official_failures.csv",
    markdown_out: str = "results/reports/final_official_report.md",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    failure_frames: list[pd.DataFrame] = []
    fig, axes = plt.subplots(1, len(OFFICIAL_PROTOCOL.benchmarks), figsize=(6 * len(OFFICIAL_PROTOCOL.benchmarks), 5), squeeze=False)

    for index, benchmark_cfg in enumerate(OFFICIAL_PROTOCOL.benchmarks):
        benchmark = benchmark_cfg.benchmark
        summary, selection, failures = load_benchmark_bundle(benchmark)
        rows.append(select_final_rows(summary, selection, benchmark))
        if not failures.empty:
            tagged = failures.copy()
            tagged.insert(0, "benchmark", benchmark)
            failure_frames.append(tagged)

        ax = axes[0][index]
        for family, group in summary.groupby("policy_family", dropna=False):
            ax.scatter(group["avg_cost"], group["accuracy"], label=str(family))
        ax.set_title(benchmark.upper())
        ax.set_xlabel("avg cost")
        ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)

    results = pd.DataFrame(rows)
    failures_df = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(columns=["benchmark", "error_type", "count"])

    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_path, index=False)

    failures_path = Path(failures_out)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    failures_df.to_csv(failures_path, index=False)

    figure_path = Path(figure_out)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path)
    plt.close(fig)

    md_path = Path(markdown_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    learned_wins = results[results["beats_strongest_fixed"].astype(bool) & results["beats_aggregate_baseline"].astype(bool)]
    lines = [
        "# ThinkRouter Final Official Report",
        "",
        f"Protocol version: `{OFFICIAL_PROTOCOL.version}`",
        "",
        "This report contains the only official benchmark results tracked by the repository.",
        "",
        results.to_markdown(index=False),
        "",
        "## Win Condition",
        "",
        f"- Benchmarks where the learned router beats both strongest fixed and aggregate baselines: `{', '.join(learned_wins['benchmark'].astype(str).tolist()) or 'none'}`",
        "",
        "## Failure Summary",
        "",
        failures_df.to_markdown(index=False) if not failures_df.empty else "No failure rows were produced.",
        "",
        "Historical smoke/dev/test20 artifacts are not part of this official report and are not kept in the public result tree.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return results, failures_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final official ThinkRouter report from per-benchmark official outputs.")
    parser.add_argument("--summary-out", default="results/tables/final_official_results.csv")
    parser.add_argument("--figure-out", default="results/figures/final_official_pareto.png")
    parser.add_argument("--failures-out", default="results/tables/final_official_failures.csv")
    parser.add_argument("--markdown-out", default="results/reports/final_official_report.md")
    args = parser.parse_args()

    results, failures = build_final_outputs(args.summary_out, args.figure_out, args.failures_out, args.markdown_out)
    print(results.to_string(index=False))
    if not failures.empty:
        print(failures.to_string(index=False))


if __name__ == "__main__":
    main()
