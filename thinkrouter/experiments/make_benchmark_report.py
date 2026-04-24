from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPORT_INPUTS = [
    ("gsm8k", "test20", "results/tables/qwen_gsm8k_official_test20_policy_summary.csv"),
    ("gsm8k", "test20", "results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_summary.csv"),
    ("math", "test20", "results/tables/qwen_math_official_test20_policy_summary.csv"),
]


def classify_policy_family(policy: str) -> str:
    if policy.startswith("phase2_"):
        return policy
    if policy.startswith("fixed_budget_"):
        return "fixed_budget"
    if policy.startswith("fixed_model_budget_"):
        return "fixed_model_budget"
    if policy.startswith("model_only_"):
        return "model_only"
    if policy.startswith("budget_only_"):
        return "budget_only"
    if policy.startswith("aggregate_utility_safe_fallback_"):
        return "joint_safe_fallback"
    if policy.startswith("aggregate_utility_"):
        return "joint_aggregate_utility"
    if policy.startswith("safe_learned_policy"):
        return "safe_learned"
    if policy.startswith("learned_policy"):
        return "learned"
    if policy == "oracle_lowest_cost_correct":
        return "oracle"
    return "other"


def load_rows() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for benchmark, split, path in REPORT_INPUTS:
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "benchmark", benchmark)
        df.insert(1, "split", split)
        rows.append(df)
    if not rows:
        raise ValueError("No benchmark report inputs were found.")
    return pd.concat(rows, ignore_index=True)


def add_relative_cost(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "policy_family" in out.columns:
        existing = out["policy_family"].astype("string")
        out["policy_family"] = existing.where(existing.notna(), out["policy"].astype(str).map(classify_policy_family))
    else:
        out["policy_family"] = out["policy"].astype(str).map(classify_policy_family)
    out["cost_vs_fixed_1024"] = pd.NA
    for (benchmark, split), group in out.groupby(["benchmark", "split"]):
        baseline = group[group["policy"] == "fixed_budget_1024"]
        if baseline.empty:
            continue
        baseline_cost = float(baseline.iloc[0]["avg_cost"])
        mask = (out["benchmark"] == benchmark) & (out["split"] == split)
        out.loc[mask, "cost_vs_fixed_1024"] = out.loc[mask, "avg_cost"].astype(float) / baseline_cost
    return out


def write_markdown(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    available_columns = [
        "benchmark",
        "split",
        "policy_family",
        "policy",
        "selected_model_provider",
        "selected_model_tier",
        "selected_model_alias",
        "accuracy",
        "avg_cost",
        "p95_latency",
        "cost_vs_fixed_1024",
        "n",
    ]
    table = df[[column for column in available_columns if column in df.columns]].copy()
    for col in ["accuracy", "avg_cost", "p95_latency", "cost_vs_fixed_1024"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(6)
    lines = [
        "# Qwen Multi-Benchmark Policy Report",
        "",
        "This report aggregates committed Qwen policy summaries across official GSM8K and MATH subsets.",
        "",
        table.to_markdown(index=False),
        "",
        "Notes:",
        "",
        "- GSM8K rows use held-out `test20` results.",
        "- MATH rows use held-out `test20` results.",
        "- `oracle_lowest_cost_correct` is an offline upper bound and is not deployable.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(summary_out: str, markdown_out: str) -> pd.DataFrame:
    df = add_relative_cost(load_rows())
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    write_markdown(df, markdown_out)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a consolidated Qwen multi-benchmark policy report.")
    parser.add_argument("--summary-out", default="results/tables/qwen_multi_benchmark_policy_report.csv")
    parser.add_argument("--markdown-out", default="results/reports/qwen_multi_benchmark_policy_report.md")
    args = parser.parse_args()

    df = build_report(args.summary_out, args.markdown_out)
    print(df.to_string(index=False))
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
