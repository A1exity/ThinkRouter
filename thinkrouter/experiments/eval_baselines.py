from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_baselines(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["selected_model", "selected_budget"], dropna=False).agg(
        accuracy=("is_correct", "mean"),
        avg_cost=("cost_usd", "mean"),
        p95_latency=("latency_s", lambda s: s.quantile(0.95)),
        n=("is_correct", "size"),
    )
    return grouped.reset_index().sort_values(["accuracy", "avg_cost"], ascending=[False, True])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline-like model/budget results from traces.")
    parser.add_argument("csv", help="Trace CSV path.")
    parser.add_argument("--out", default="results/tables/baseline_summary.csv")
    args = parser.parse_args()
    summary = summarize_baselines(args.csv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
