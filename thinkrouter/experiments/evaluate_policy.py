from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from thinkrouter.experiments.analyze_failures import parse_metadata


@dataclass(frozen=True)
class UtilityWeights:
    alpha: float = 1.0
    beta: float = 5.0
    gamma: float = 0.02


def add_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        metadata_values = out["metadata"] if "metadata" in out.columns else [None] * len(out)
        row_ids = out["id"] if "id" in out.columns else range(len(out))
        out["sample_id"] = [parse_metadata(value).get("sample_id", str(row_id)) for value, row_id in zip(metadata_values, row_ids)]
    return out


def utility(row: pd.Series, weights: UtilityWeights) -> float:
    accuracy = float(row.get("accuracy", row.get("is_correct", 0.0)))
    cost = float(row.get("avg_cost", row.get("cost_usd", 0.0)))
    latency = float(row.get("avg_latency", row.get("latency_s", 0.0)))
    return weights.alpha * accuracy - weights.beta * cost - weights.gamma * latency


def summarize_selection(name: str, selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"policy": name, "accuracy": 0.0, "avg_cost": 0.0, "total_cost": 0.0, "avg_latency": 0.0, "p95_latency": 0.0, "n": 0}
    return {
        "policy": name,
        "accuracy": float(selected["is_correct"].mean()),
        "avg_cost": float(selected["cost_usd"].mean()),
        "total_cost": float(selected["cost_usd"].sum()),
        "avg_latency": float(selected["latency_s"].mean()),
        "p95_latency": float(selected["latency_s"].quantile(0.95)),
        "n": int(len(selected)),
    }


def fixed_model_budget_policy(df: pd.DataFrame, model_id: str, budget: int) -> pd.DataFrame:
    return df[(df["selected_model"].astype(str) == str(model_id)) & (df["selected_budget"].astype(int) == int(budget))].copy()


def fixed_budget_policy(df: pd.DataFrame, budget: int) -> pd.DataFrame:
    return df[df["selected_budget"].astype(int) == budget].copy()


def oracle_lowest_cost_correct(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in df.groupby("sample_id", sort=False):
        correct = group[group["is_correct"].astype(bool)]
        candidates = correct if not correct.empty else group
        rows.append(candidates.sort_values(["is_correct", "cost_usd", "latency_s", "selected_budget"], ascending=[False, True, True, True]).iloc[0])
    return pd.DataFrame(rows)


def aggregate_utility_policy(df: pd.DataFrame, weights: UtilityWeights) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = df.groupby(["selected_model", "selected_budget"], dropna=False).agg(
        accuracy=("is_correct", "mean"),
        avg_cost=("cost_usd", "mean"),
        avg_latency=("latency_s", "mean"),
        p95_latency=("latency_s", lambda s: s.quantile(0.95)),
        n=("is_correct", "size"),
    ).reset_index()
    stats["utility"] = [utility(row, weights) for _, row in stats.iterrows()]
    best = stats.sort_values(["utility", "accuracy", "avg_cost"], ascending=[False, False, True]).iloc[0]
    selected = df[(df["selected_model"] == best["selected_model"]) & (df["selected_budget"].astype(int) == int(best["selected_budget"]))].copy()
    selected["policy_selected_utility"] = float(best["utility"])
    return selected, stats


def best_model_only_policy(df: pd.DataFrame, weights: UtilityWeights) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats = df.groupby("selected_model", dropna=False).agg(
        accuracy=("is_correct", "mean"),
        avg_cost=("cost_usd", "mean"),
        avg_latency=("latency_s", "mean"),
    ).reset_index()
    stats["utility"] = [utility(row, weights) for _, row in stats.iterrows()]
    best = stats.sort_values(["utility", "accuracy", "avg_cost"], ascending=[False, False, True]).iloc[0]
    selected = df[df["selected_model"].astype(str) == str(best["selected_model"])].copy()
    return selected, {"selected_model": str(best["selected_model"]), "utility": float(best["utility"])}


def best_budget_only_policy(df: pd.DataFrame, weights: UtilityWeights) -> tuple[pd.DataFrame, dict[str, Any]]:
    stats = df.groupby("selected_budget", dropna=False).agg(
        accuracy=("is_correct", "mean"),
        avg_cost=("cost_usd", "mean"),
        avg_latency=("latency_s", "mean"),
    ).reset_index()
    stats["utility"] = [utility(row, weights) for _, row in stats.iterrows()]
    best = stats.sort_values(["utility", "accuracy", "avg_cost"], ascending=[False, False, True]).iloc[0]
    selected = df[df["selected_budget"].astype(int) == int(best["selected_budget"])].copy()
    return selected, {"selected_budget": int(best["selected_budget"]), "utility": float(best["utility"])}


def evaluate_policies(csv_path: str, weights: UtilityWeights | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    weights = weights or UtilityWeights()
    df = add_sample_id(pd.read_csv(csv_path))
    rows: list[dict[str, Any]] = []
    for model_id in sorted(df["selected_model"].astype(str).unique()):
        for budget in sorted(df.loc[df["selected_model"].astype(str) == model_id, "selected_budget"].astype(int).unique()):
            rows.append(summarize_selection(f"fixed_{model_id}_budget_{budget}", fixed_model_budget_policy(df, model_id, budget)))
    for budget in sorted(df["selected_budget"].astype(int).unique()):
        rows.append(summarize_selection(f"fixed_budget_{budget}", fixed_budget_policy(df, budget)))
    rows.append(summarize_selection("oracle_lowest_cost_correct", oracle_lowest_cost_correct(df)))
    model_only_selected, model_only_meta = best_model_only_policy(df, weights)
    rows.append(summarize_selection(f"model_only_best_{model_only_meta['selected_model']}", model_only_selected))
    budget_only_selected, budget_only_meta = best_budget_only_policy(df, weights)
    rows.append(summarize_selection(f"budget_only_best_{budget_only_meta['selected_budget']}", budget_only_selected))
    aggregate_selected, aggregate_stats = aggregate_utility_policy(df, weights)
    budget = int(aggregate_selected["selected_budget"].iloc[0]) if not aggregate_selected.empty else -1
    rows.append(summarize_selection(f"aggregate_utility_budget_{budget}", aggregate_selected))
    return pd.DataFrame(rows), aggregate_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline routing policies from an existing grid CSV.")
    parser.add_argument("csv", help="Grid CSV path containing all candidate model/budget traces.")
    parser.add_argument("--out", default="results/tables/policy_summary.csv", help="Policy summary output CSV.")
    parser.add_argument("--stats-out", default=None, help="Optional aggregate utility stats output CSV.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    args = parser.parse_args()

    summary, stats = evaluate_policies(args.csv, UtilityWeights(alpha=args.alpha, beta=args.beta, gamma=args.gamma))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_path}")
    if args.stats_out:
        stats_path = Path(args.stats_out)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(stats_path, index=False)
        print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
