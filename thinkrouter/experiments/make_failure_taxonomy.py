from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thinkrouter.experiments.analyze_failures import analyze_failures


def summarize_failure_taxonomy(csv_path: str) -> pd.DataFrame:
    failures = analyze_failures(csv_path)
    if failures.empty:
        return pd.DataFrame(columns=["error_type", "count", "avg_cost", "avg_latency"])
    summary = (
        failures.groupby("error_type", dropna=False)
        .agg(
            count=("error_type", "size"),
            avg_cost=("cost_usd", "mean"),
            avg_latency=("latency_s", "mean"),
        )
        .reset_index()
        .sort_values(["count", "avg_cost"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return summary


def build_failure_taxonomy_report(csv_path: str, summary_out: str, markdown_out: str) -> pd.DataFrame:
    summary = summarize_failure_taxonomy(csv_path)
    out_csv = Path(summary_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    lines = [
        "# ThinkRouter Failure Taxonomy Report",
        "",
        f"Source: `{Path(csv_path).name}`",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No failures.",
        "",
    ]
    out_md = Path(markdown_out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize failure taxonomy from a completed grid CSV.")
    parser.add_argument("csv", help="Grid CSV path.")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--markdown-out", required=True)
    args = parser.parse_args()

    summary = build_failure_taxonomy_report(args.csv, args.summary_out, args.markdown_out)
    print(summary.to_string(index=False) if not summary.empty else "No failures.")
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
