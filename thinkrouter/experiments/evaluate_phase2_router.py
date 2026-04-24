from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thinkrouter.adapters import default_model_configs
from thinkrouter.experiments.evaluate_policy import add_sample_id, summarize_selection
from thinkrouter.routers import LogRegJointRouter, MLPFactorizedRouter, ThresholdRouter, UncertaintyAwareRouter, load_factorized_artifact, load_logreg_joint_artifact


def replay_router(csv_path: str, router_name: str, model_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = add_sample_id(pd.read_csv(csv_path))
    models = list(default_model_configs().values())
    if router_name == "threshold":
        router = ThresholdRouter(models)
    elif router_name == "logreg_joint":
        if not model_path:
            raise ValueError("--model is required for logreg_joint")
        router = LogRegJointRouter(models, artifact=load_logreg_joint_artifact(model_path))
    elif router_name == "mlp_factorized":
        if not model_path:
            raise ValueError("--model is required for mlp_factorized")
        router = MLPFactorizedRouter(models, artifact=load_factorized_artifact(model_path))
    elif router_name == "uncertainty_aware":
        if not model_path:
            raise ValueError("--model is required for uncertainty_aware")
        router = UncertaintyAwareRouter(models, artifact=load_factorized_artifact(model_path))
    else:
        raise ValueError(f"Unsupported router: {router_name}")

    selected_rows: list[pd.Series] = []
    for sample_id, group in df.groupby("sample_id", sort=False):
        first = group.iloc[0]
        decision = router.route(str(first["query"]), str(first["task_type"]))
        exact = group[
            (group["selected_model"].astype(str) == decision.model_id)
            & (group["selected_budget"].astype(int) == int(decision.budget))
        ]
        candidates = exact if not exact.empty else group.sort_values(["cost_usd", "latency_s", "selected_budget"], ascending=[True, True, True])
        selected = candidates.iloc[0].copy()
        selected["policy"] = router_name
        selected["route_confidence"] = decision.route_confidence
        selected["fallback_triggered"] = decision.fallback_triggered
        selected["fallback_reason"] = decision.fallback_reason
        selected["router_name"] = decision.router_name
        selected_rows.append(selected)
    selected_df = pd.DataFrame(selected_rows)
    summary = pd.DataFrame([summarize_selection(router_name, selected_df)])
    return summary, selected_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a Phase 2 router artifact on a completed grid CSV.")
    parser.add_argument("csv", help="Grid CSV path.")
    parser.add_argument("--router", choices=["threshold", "logreg_joint", "mlp_factorized", "uncertainty_aware"], required=True)
    parser.add_argument("--model", default=None, help="Artifact path for learned routers.")
    parser.add_argument("--out", required=True, help="Summary CSV output path.")
    parser.add_argument("--selected-out", default=None, help="Optional selected-trace CSV output path.")
    args = parser.parse_args()

    summary, selected = replay_router(args.csv, args.router, model_path=args.model)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out}")
    if args.selected_out:
        selected_out = Path(args.selected_out)
        selected_out.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(selected_out, index=False)
        print(f"Wrote {selected_out}")


if __name__ == "__main__":
    main()
