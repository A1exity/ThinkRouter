from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from thinkrouter.experiments.eval_baselines import summarize_baselines


FAMILY_STYLES = {
    "fixed_model_budget": {"color": "#94A3B8", "marker": "o", "size": 45},
    "model_only": {"color": "#2563EB", "marker": "X", "size": 110},
    "budget_only": {"color": "#059669", "marker": "^", "size": 110},
    "joint_aggregate_utility": {"color": "#DC2626", "marker": "*", "size": 180},
    "joint_safe_fallback": {"color": "#D97706", "marker": "P", "size": 150},
}


def make_pareto_plot(csv_path: str, out_path: str, phase2_routers: list[str] | None = None) -> None:
    summary = summarize_baselines(csv_path, phase2_routers=phase2_routers)
    if summary.empty:
        raise ValueError("Input CSV is empty.")
    fig, ax = plt.subplots(figsize=(7, 5))
    for family, group in summary.groupby("policy_family", dropna=False):
        style = FAMILY_STYLES.get(str(family), {"color": "#111827", "marker": "o", "size": 60})
        ax.scatter(group["avg_cost"], group["accuracy"], c=style["color"], marker=style["marker"], s=style["size"], label=str(family))
        for row in group.itertuples(index=False):
            if str(family) == "fixed_model_budget":
                tier = "" if not getattr(row, "selected_model_tier", None) or str(getattr(row, "selected_model_tier")) == "nan" else f"[{row.selected_model_tier}]"
                label = f"{row.selected_model}{tier}/{int(row.selected_budget)}"
            else:
                label = str(row.policy)
            ax.annotate(label, (row.avg_cost, row.accuracy), fontsize=8)
    ax.set_xlabel("Average cost per query (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_title("ThinkRouter Cost-Accuracy Pareto")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ThinkRouter plots from trace CSV.")
    parser.add_argument("csv", help="Trace CSV path.")
    parser.add_argument("--out", default="results/figures/pareto.png")
    parser.add_argument(
        "--phase2-router",
        action="append",
        default=[],
        help="Optional Phase 2 router replay spec. Use NAME or NAME=artifact_path.",
    )
    args = parser.parse_args()
    make_pareto_plot(args.csv, args.out, phase2_routers=args.phase2_router)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
