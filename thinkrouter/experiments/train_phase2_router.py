from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thinkrouter.routers import save_factorized_artifact, save_logreg_joint_artifact, train_factorized_router, train_logreg_joint_router
from thinkrouter.training import UtilityObjective, derive_factorized_examples, derive_joint_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Phase 2 router artifact from a completed grid CSV.")
    parser.add_argument("csv", help="Grid CSV path containing candidate traces.")
    parser.add_argument("--router", choices=["logreg_joint", "mlp_factorized"], required=True)
    parser.add_argument("--out", required=True, help="Output artifact path.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    objective = UtilityObjective(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    if args.router == "logreg_joint":
        artifact = train_logreg_joint_router(derive_joint_examples(df, objective))
        save_logreg_joint_artifact(artifact, args.out)
    else:
        artifact = train_factorized_router(derive_factorized_examples(df, objective))
        save_factorized_artifact(artifact, args.out)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
