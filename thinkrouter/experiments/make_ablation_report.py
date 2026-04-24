from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def classify_ablation(policy_family: str) -> str:
    family = str(policy_family)
    if family == "phase2_threshold":
        return "surface_only_threshold"
    if family == "phase2_logreg_joint":
        return "joint_linear_router"
    if family == "phase2_mlp_factorized":
        return "factorized_router"
    if family == "phase2_uncertainty_aware":
        return "factorized_plus_uncertainty"
    if family == "joint_aggregate_utility":
        return "aggregate_utility_baseline"
    if family == "joint_safe_fallback":
        return "aggregate_safe_fallback"
    if family == "model_only":
        return "model_only_baseline"
    if family == "budget_only":
        return "budget_only_baseline"
    if family == "fixed_model_budget":
        return "fixed_model_budget"
    return family


def build_ablation_report(summary_csvs: list[str], summary_out: str, markdown_out: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in summary_csvs:
        csv_path = Path(path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df.insert(0, "source_file", csv_path.name)
        rows.append(df)
    if not rows:
        raise ValueError("No ablation inputs were found.")
    combined = pd.concat(rows, ignore_index=True)
    combined["ablation_group"] = combined["policy_family"].astype(str).map(classify_ablation)
    columns = [column for column in ["source_file", "policy", "policy_family", "ablation_group", "accuracy", "avg_cost", "avg_latency", "utility"] if column in combined.columns]
    report = combined[columns].copy()
    for metric in ["accuracy", "avg_cost", "avg_latency", "utility"]:
        if metric in report.columns:
            report[metric] = pd.to_numeric(report[metric], errors="coerce").round(6)
    out_csv = Path(summary_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_csv, index=False)

    lines = [
        "# ThinkRouter Ablation Report",
        "",
        "This report groups integrated policy outputs into coarse ablation buckets.",
        "",
        report.to_markdown(index=False),
        "",
    ]
    out_md = Path(markdown_out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Phase 4 ablation report from integrated summary CSVs.")
    parser.add_argument("summary_csv", nargs="+", help="One or more integrated summary CSV paths.")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()

    report = build_ablation_report(args.summary_csv, args.summary_out, args.markdown_out)
    print(report.to_string(index=False))
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
