from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_pareto_plot(csv_path: str, out_path: str) -> None:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    summary = df.groupby(["selected_model", "selected_budget"], dropna=False).agg(
        accuracy=("is_correct", "mean"),
        avg_cost=("cost_usd", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(summary["avg_cost"], summary["accuracy"])
    for row in summary.itertuples(index=False):
        ax.annotate(f"{row.selected_model}/{row.selected_budget}", (row.avg_cost, row.accuracy), fontsize=8)
    ax.set_xlabel("Average cost per query (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_title("ThinkRouter Cost-Accuracy Pareto")
    ax.grid(True, alpha=0.3)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ThinkRouter plots from trace CSV.")
    parser.add_argument("csv", help="Trace CSV path.")
    parser.add_argument("--out", default="results/figures/pareto.png")
    args = parser.parse_args()
    make_pareto_plot(args.csv, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
