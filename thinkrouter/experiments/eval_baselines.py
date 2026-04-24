from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thinkrouter.experiments.evaluate_policy import UtilityWeights, add_sample_id, aggregate_utility_policy, best_budget_only_policy, best_model_only_policy, safe_fallback_policy, summarize_selection
from thinkrouter.experiments.phase2_router_replay import replay_router_specs


def summarize_baselines(
    csv_path: str,
    weights: UtilityWeights | None = None,
    phase2_routers: list[str] | None = None,
) -> pd.DataFrame:
    weights = weights or UtilityWeights()
    df = add_sample_id(pd.read_csv(csv_path))
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []

    grouped = (
        df.groupby(["selected_model", "selected_budget"], dropna=False)
        .agg(
            accuracy=("is_correct", "mean"),
            avg_cost=("cost_usd", "mean"),
            total_cost=("cost_usd", "sum"),
            avg_latency=("latency_s", "mean"),
            p95_latency=("latency_s", lambda s: s.quantile(0.95)),
            n=("is_correct", "size"),
        )
        .reset_index()
    )
    for row in grouped.itertuples(index=False):
        rows.append(
            {
                "policy": f"fixed_model_budget_{row.selected_model}_{int(row.selected_budget)}",
                "policy_family": "fixed_model_budget",
                "selected_model": row.selected_model,
                "selected_model_provider": df.loc[df["selected_model"].astype(str) == str(row.selected_model), "selected_model_provider"].dropna().astype(str).iloc[0]
                if "selected_model_provider" in df.columns and not df.loc[df["selected_model"].astype(str) == str(row.selected_model), "selected_model_provider"].dropna().empty
                else pd.NA,
                "selected_model_tier": df.loc[df["selected_model"].astype(str) == str(row.selected_model), "selected_model_tier"].dropna().astype(str).iloc[0]
                if "selected_model_tier" in df.columns and not df.loc[df["selected_model"].astype(str) == str(row.selected_model), "selected_model_tier"].dropna().empty
                else pd.NA,
                "selected_model_alias": df.loc[df["selected_model"].astype(str) == str(row.selected_model), "selected_model_alias"].dropna().astype(str).iloc[0]
                if "selected_model_alias" in df.columns and not df.loc[df["selected_model"].astype(str) == str(row.selected_model), "selected_model_alias"].dropna().empty
                else pd.NA,
                "selected_budget": int(row.selected_budget),
                "accuracy": float(row.accuracy),
                "avg_cost": float(row.avg_cost),
                "total_cost": float(row.total_cost),
                "avg_latency": float(row.avg_latency),
                "p95_latency": float(row.p95_latency),
                "n": int(row.n),
            }
        )

    model_only_selected, model_only_meta = best_model_only_policy(df, weights)
    rows.append(
        {
            **summarize_selection(f"model_only_best_{model_only_meta['selected_model']}", model_only_selected),
            "policy_family": "model_only",
            "selected_model": model_only_meta["selected_model"],
            "selected_model_provider": _first_non_null(model_only_selected, "selected_model_provider"),
            "selected_model_tier": _first_non_null(model_only_selected, "selected_model_tier"),
            "selected_model_alias": _first_non_null(model_only_selected, "selected_model_alias"),
            "selected_budget": pd.NA,
        }
    )

    budget_only_selected, budget_only_meta = best_budget_only_policy(df, weights)
    rows.append(
        {
            **summarize_selection(f"budget_only_best_{budget_only_meta['selected_budget']}", budget_only_selected),
            "policy_family": "budget_only",
            "selected_model": pd.NA,
            "selected_model_provider": pd.NA,
            "selected_model_tier": pd.NA,
            "selected_model_alias": pd.NA,
            "selected_budget": int(budget_only_meta["selected_budget"]),
        }
    )

    aggregate_selected, _ = aggregate_utility_policy(df, weights)
    aggregate_budget = int(aggregate_selected["selected_budget"].iloc[0]) if not aggregate_selected.empty else -1
    aggregate_model = str(aggregate_selected["selected_model"].iloc[0]) if not aggregate_selected.empty else ""
    rows.append(
        {
            **summarize_selection(f"aggregate_utility_{aggregate_model}_{aggregate_budget}", aggregate_selected),
            "policy_family": "joint_aggregate_utility",
            "selected_model": aggregate_model,
            "selected_model_provider": _first_non_null(aggregate_selected, "selected_model_provider"),
            "selected_model_tier": _first_non_null(aggregate_selected, "selected_model_tier"),
            "selected_model_alias": _first_non_null(aggregate_selected, "selected_model_alias"),
            "selected_budget": aggregate_budget,
        }
    )
    safe_selected = safe_fallback_policy(df, aggregate_selected, int(budget_only_meta["selected_budget"]))
    rows.append(
        {
            **summarize_selection(
                f"aggregate_utility_safe_fallback_{aggregate_model}_{aggregate_budget}_to_{budget_only_meta['selected_budget']}",
                safe_selected,
            ),
            "policy_family": "joint_safe_fallback",
            "selected_model": aggregate_model,
            "selected_model_provider": _first_non_null(safe_selected, "selected_model_provider"),
            "selected_model_tier": _first_non_null(safe_selected, "selected_model_tier"),
            "selected_model_alias": _first_non_null(safe_selected, "selected_model_alias"),
            "selected_budget": aggregate_budget,
        }
    )

    summary = pd.DataFrame(rows)
    if phase2_routers:
        phase2_summary, _ = replay_router_specs(csv_path, phase2_routers)
        if not phase2_summary.empty:
            summary = pd.concat([summary, phase2_summary], ignore_index=True)
    return summary.sort_values(["policy_family", "accuracy", "avg_cost"], ascending=[True, False, True]).reset_index(drop=True)


def _first_non_null(df: pd.DataFrame, column: str) -> object:
    if column not in df.columns:
        return pd.NA
    values = df[column].dropna()
    if values.empty:
        return pd.NA
    return values.astype(str).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize baseline-like model/budget results from traces.")
    parser.add_argument("csv", help="Trace CSV path.")
    parser.add_argument("--out", default="results/tables/baseline_summary.csv")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    parser.add_argument(
        "--phase2-router",
        action="append",
        default=[],
        help="Optional Phase 2 router replay spec. Use NAME or NAME=artifact_path.",
    )
    args = parser.parse_args()
    summary = summarize_baselines(
        args.csv,
        UtilityWeights(alpha=args.alpha, beta=args.beta, gamma=args.gamma),
        phase2_routers=args.phase2_router,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
