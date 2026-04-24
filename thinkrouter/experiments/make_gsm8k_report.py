from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPORT_INPUTS = [
    ("dev20", "fixed_or_oracle", "results/tables/qwen_gsm8k_official_dev20_policy_summary.csv"),
    ("test20", "fixed_or_oracle", "results/tables/qwen_gsm8k_official_test20_policy_summary.csv"),
    ("dev20", "raw_learned", "results/tables/qwen_gsm8k_official_train60_to_dev20_raw_learned_policy_summary.csv"),
    ("dev20", "safe_train_only", "results/tables/qwen_gsm8k_official_train60_to_dev20_safe_policy_summary.csv"),
    ("test20", "raw_learned", "results/tables/qwen_gsm8k_official_train60_to_test20_raw_learned_policy_summary.csv"),
    ("test20", "safe_train_only", "results/tables/qwen_gsm8k_official_train60_to_test20_safe_policy_summary.csv"),
    ("test20", "dev_calibrated", "results/tables/qwen_gsm8k_official_train60_dev20_to_test20_calibrated_policy_summary.csv"),
]


def classify_policy_family(policy: str) -> str:
    if policy.startswith("fixed_budget_"):
        return "fixed_budget"
    if policy.startswith("fixed_model_budget_"):
        return "fixed_model_budget"
    if policy.startswith("model_only_"):
        return "model_only"
    if policy.startswith("budget_only_"):
        return "budget_only"
    if policy.startswith("safe_learned_policy"):
        return "safe_learned"
    if policy.startswith("aggregate_utility_safe_fallback_"):
        return "joint_safe_fallback"
    if policy.startswith("aggregate_utility_"):
        return "joint_aggregate_utility"
    if policy.startswith("learned_policy"):
        return "learned"
    if policy == "oracle_lowest_cost_correct":
        return "oracle"
    return "other"


def load_policy_rows(paths: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split, source, path in paths:
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df.insert(0, "split", split)
        df.insert(1, "source", source)
        rows.append(df)
    if not rows:
        raise ValueError("No report inputs were found.")
    return pd.concat(rows, ignore_index=True)


def add_relative_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["policy_family"] = out["policy"].astype(str).map(classify_policy_family)
    out["cost_vs_fixed_1024"] = pd.NA
    out["accuracy_delta_vs_fixed_1024"] = pd.NA
    for split, group in out.groupby("split"):
        baseline = group[group["policy"] == "fixed_budget_1024"]
        if baseline.empty:
            continue
        baseline_row = baseline.iloc[0]
        mask = out["split"] == split
        out.loc[mask, "cost_vs_fixed_1024"] = out.loc[mask, "avg_cost"].astype(float) / float(baseline_row["avg_cost"])
        out.loc[mask, "accuracy_delta_vs_fixed_1024"] = out.loc[mask, "accuracy"].astype(float) - float(baseline_row["accuracy"])
    return out


def write_markdown_report(df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "source",
        "policy_family",
        "policy",
        "selected_model_provider",
        "selected_model_tier",
        "selected_model_alias",
        "accuracy",
        "avg_cost",
        "p95_latency",
        "cost_vs_fixed_1024",
        "accuracy_delta_vs_fixed_1024",
        "n",
    ]
    table = df[[column for column in columns if column in df.columns]].copy()
    for col in ["accuracy", "avg_cost", "p95_latency", "cost_vs_fixed_1024", "accuracy_delta_vs_fixed_1024"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").round(6)
    lines = [
        "# GSM8K Qwen Report",
        "",
        "This report aggregates the committed Qwen GSM8K dev/test policy results.",
        "",
        table.to_markdown(index=False),
        "",
        "Interpretation:",
        "",
        "- `fixed_budget_1024` is the high-budget reference for cost and accuracy deltas.",
        "- `oracle_lowest_cost_correct` is an offline upper bound and is not deployable.",
        "- `safe_learned_policy_fallback_budget_256` is trained on train60, calibrated on dev20, and evaluated on held-out test20.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_test_policy_comparison(df: pd.DataFrame, out_path: str) -> None:
    test = df[df["split"] == "test20"].copy()
    if test.empty:
        raise ValueError("No test20 rows available for plotting.")
    order = [
        "fixed_budget_0",
        "fixed_budget_256",
        "fixed_budget_1024",
        "budget_only_best_0",
        "model_only_best_mock-cheap",
        "learned_policy",
        "safe_learned_policy_fallback_budget_0",
        "safe_learned_policy_fallback_budget_256",
        "aggregate_utility_mock-cheap_0",
        "oracle_lowest_cost_correct",
    ]
    test["policy_order"] = test["policy"].apply(lambda value: order.index(value) if value in order else len(order))
    test = test.sort_values(["policy_order", "source"])
    labels = [str(value).replace("safe_learned_policy_", "safe_").replace("oracle_lowest_cost_correct", "oracle") for value in test["policy"]]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = range(len(test))
    ax1.bar(x, test["accuracy"], color="#3B82F6", alpha=0.75, label="accuracy")
    ax1.set_ylim(0.8, 1.02)
    ax1.set_ylabel("Accuracy")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(list(x), test["avg_cost"], color="#DC2626", marker="o", linewidth=2, label="avg cost")
    ax2.set_ylabel("Average cost per query (USD)")

    ax1.set_title("Held-out GSM8K test20 policy comparison")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def build_report(summary_out: str, markdown_out: str, figure_out: str) -> pd.DataFrame:
    df = add_relative_metrics(load_policy_rows(REPORT_INPUTS))
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)
    write_markdown_report(df, markdown_out)
    plot_test_policy_comparison(df, figure_out)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a consolidated GSM8K Qwen report from committed result CSVs.")
    parser.add_argument("--summary-out", default="results/tables/qwen_gsm8k_final_policy_report.csv")
    parser.add_argument("--markdown-out", default="results/reports/qwen_gsm8k_final_policy_report.md")
    parser.add_argument("--figure-out", default="results/figures/qwen_gsm8k_test20_policy_comparison.png")
    args = parser.parse_args()

    df = build_report(args.summary_out, args.markdown_out, args.figure_out)
    print(df.to_string(index=False))
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.markdown_out}")
    print(f"Wrote {args.figure_out}")


if __name__ == "__main__":
    main()
