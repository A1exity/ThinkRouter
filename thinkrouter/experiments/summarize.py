from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thinkrouter.analytics.stability import bootstrap_metric_ci


def summarize_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    if "policy" in df.columns and "accuracy" in df.columns:
        for _, row in df.iterrows():
            rows.append(
                {
                    "policy": row.get("policy"),
                    "accuracy": row.get("accuracy"),
                    "avg_cost": row.get("avg_cost"),
                    "avg_latency": row.get("avg_latency", row.get("p95_latency")),
                }
            )
        return pd.DataFrame(rows)

    group_columns = [column for column in ["task_type", "selected_model", "selected_budget"] if column in df.columns]
    grouped = df.groupby(group_columns, dropna=False) if group_columns else [(("all",), df)]
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        ci = bootstrap_metric_ci(group["is_correct"].astype(float))
        rows.append(
            {
                **{column: value for column, value in zip(group_columns, key)},
                "n": int(len(group)),
                "accuracy_mean": ci["mean"],
                "accuracy_ci_lower": ci["lower"],
                "accuracy_ci_upper": ci["upper"],
                "avg_cost": float(pd.to_numeric(group["cost_usd"], errors="coerce").mean()),
                "avg_latency": float(pd.to_numeric(group["latency_s"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a ThinkRouter CSV with optional bootstrap confidence intervals.")
    parser.add_argument("csv", help="Input CSV path.")
    parser.add_argument("--out", required=True, help="Summary CSV output path.")
    args = parser.parse_args()

    summary = summarize_csv(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False) if not summary.empty else "Empty summary.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
