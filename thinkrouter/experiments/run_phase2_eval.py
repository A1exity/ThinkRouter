from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thinkrouter.experiments.eval_baselines import summarize_baselines
from thinkrouter.experiments.evaluate_phase2_router import replay_router
from thinkrouter.experiments.make_plots import make_pareto_plot
from thinkrouter.routers import save_factorized_artifact, save_logreg_joint_artifact, train_factorized_router, train_logreg_joint_router
from thinkrouter.training import UtilityObjective, derive_factorized_examples, derive_joint_examples


def run_phase2_eval(
    csv_path: str,
    out_prefix: str,
    alpha: float = 1.0,
    beta: float = 5.0,
    gamma: float = 0.02,
) -> dict[str, str]:
    df = pd.read_csv(csv_path)
    objective = UtilityObjective(alpha=alpha, beta=beta, gamma=gamma)

    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    logreg_path = Path(f"{out_prefix}_logreg_joint.joblib")
    factorized_path = Path(f"{out_prefix}_mlp_factorized.joblib")
    threshold_summary_path = Path(f"{out_prefix}_threshold_summary.csv")
    threshold_selected_path = Path(f"{out_prefix}_threshold_selected.csv")
    logreg_summary_path = Path(f"{out_prefix}_logreg_joint_summary.csv")
    logreg_selected_path = Path(f"{out_prefix}_logreg_joint_selected.csv")
    mlp_summary_path = Path(f"{out_prefix}_mlp_factorized_summary.csv")
    mlp_selected_path = Path(f"{out_prefix}_mlp_factorized_selected.csv")
    uncertainty_summary_path = Path(f"{out_prefix}_uncertainty_aware_summary.csv")
    uncertainty_selected_path = Path(f"{out_prefix}_uncertainty_aware_selected.csv")
    integrated_summary_path = Path(f"{out_prefix}_baseline_phase2_summary.csv")
    figure_path = Path(f"{out_prefix}_phase2_pareto.png")

    logreg_artifact = train_logreg_joint_router(derive_joint_examples(df, objective))
    factorized_artifact = train_factorized_router(derive_factorized_examples(df, objective))
    save_logreg_joint_artifact(logreg_artifact, logreg_path)
    save_factorized_artifact(factorized_artifact, factorized_path)

    threshold_summary, threshold_selected = replay_router(csv_path, "threshold")
    threshold_summary.to_csv(threshold_summary_path, index=False)
    threshold_selected.to_csv(threshold_selected_path, index=False)

    logreg_summary, logreg_selected = replay_router(csv_path, "logreg_joint", model_path=str(logreg_path))
    logreg_summary.to_csv(logreg_summary_path, index=False)
    logreg_selected.to_csv(logreg_selected_path, index=False)

    mlp_summary, mlp_selected = replay_router(csv_path, "mlp_factorized", model_path=str(factorized_path))
    mlp_summary.to_csv(mlp_summary_path, index=False)
    mlp_selected.to_csv(mlp_selected_path, index=False)

    uncertainty_summary, uncertainty_selected = replay_router(csv_path, "uncertainty_aware", model_path=str(factorized_path))
    uncertainty_summary.to_csv(uncertainty_summary_path, index=False)
    uncertainty_selected.to_csv(uncertainty_selected_path, index=False)

    integrated_summary = summarize_baselines(
        csv_path,
        phase2_routers=[
            "threshold",
            f"logreg_joint={logreg_path}",
            f"mlp_factorized={factorized_path}",
            f"uncertainty_aware={factorized_path}",
        ],
    )
    integrated_summary.to_csv(integrated_summary_path, index=False)

    make_pareto_plot(
        csv_path,
        str(figure_path),
        phase2_routers=[
            "threshold",
            f"logreg_joint={logreg_path}",
            f"mlp_factorized={factorized_path}",
            f"uncertainty_aware={factorized_path}",
        ],
    )

    return {
        "logreg_artifact": str(logreg_path),
        "factorized_artifact": str(factorized_path),
        "threshold_summary": str(threshold_summary_path),
        "threshold_selected": str(threshold_selected_path),
        "logreg_summary": str(logreg_summary_path),
        "logreg_selected": str(logreg_selected_path),
        "mlp_summary": str(mlp_summary_path),
        "mlp_selected": str(mlp_selected_path),
        "uncertainty_summary": str(uncertainty_summary_path),
        "uncertainty_selected": str(uncertainty_selected_path),
        "integrated_summary": str(integrated_summary_path),
        "pareto_figure": str(figure_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and replay the Phase 2 router stack on a completed grid CSV.")
    parser.add_argument("csv", help="Grid CSV path.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix without extension, for example results/qwen35_pool_gsm8k_dev10")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    args = parser.parse_args()

    outputs = run_phase2_eval(args.csv, args.out_prefix, alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
