from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_phase2_utility(df: pd.DataFrame, alpha: float = 1.0, beta: float = 5.0, gamma: float = 0.02) -> pd.DataFrame:
    out = df.copy()
    out["utility"] = (
        alpha * pd.to_numeric(out["accuracy"], errors="coerce").fillna(0.0)
        - beta * pd.to_numeric(out["avg_cost"], errors="coerce").fillna(0.0)
        - gamma * pd.to_numeric(out["avg_latency"], errors="coerce").fillna(0.0)
    )
    return out


def build_phase2_report(summary_csv: str, summary_out: str, markdown_out: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError("Phase 2 summary is empty.")
    enriched = add_phase2_utility(df)
    ranked = enriched.sort_values(["utility", "accuracy", "avg_cost"], ascending=[False, False, True]).reset_index(drop=True)
    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(summary_path, index=False)

    available_columns = [
        "policy",
        "policy_family",
        "router_name",
        "selected_model",
        "selected_model_tier",
        "selected_budget",
        "accuracy",
        "avg_cost",
        "avg_latency",
        "avg_route_confidence",
        "fallback_rate",
        "utility",
    ]
    table = ranked[[column for column in available_columns if column in ranked.columns]].copy()
    for col in ["accuracy", "avg_cost", "avg_latency", "avg_route_confidence", "fallback_rate", "utility"]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce").round(6)

    phase2_rows = ranked[ranked["policy_family"].astype(str).str.startswith("phase2_")]
    best_phase2 = phase2_rows.iloc[0] if not phase2_rows.empty else None
    best_overall = ranked.iloc[0]

    lines = [
        "# ThinkRouter Phase 2 Report",
        "",
        "This report ranks the integrated baseline and Phase 2 router summaries by utility.",
        "",
        f"- Best overall policy: `{best_overall['policy']}`",
        f"- Best overall utility: `{float(best_overall['utility']):.6f}`",
    ]
    if best_phase2 is not None:
        lines.extend(
            [
                f"- Best Phase 2 policy: `{best_phase2['policy']}`",
                f"- Best Phase 2 utility: `{float(best_phase2['utility']):.6f}`",
            ]
        )
    lines.extend(["", table.to_markdown(index=False), ""])

    out_path = Path(markdown_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a utility-ranked Phase 2 report from an integrated summary CSV.")
    parser.add_argument("summary_csv", help="Integrated baseline + Phase 2 summary CSV.")
    parser.add_argument("--summary-out", required=True, help="Ranked CSV output path.")
    parser.add_argument("--markdown-out", required=True, help="Markdown report output path.")
    args = parser.parse_args()

    df = build_phase2_report(args.summary_csv, args.summary_out, args.markdown_out)
    print(df.to_string(index=False))
    print(f"Wrote {args.summary_out}")
    print(f"Wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
