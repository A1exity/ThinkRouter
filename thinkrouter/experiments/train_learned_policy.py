from __future__ import annotations

import argparse
from pathlib import Path

from thinkrouter.experiments.evaluate_policy import UtilityWeights
from thinkrouter.experiments.learned_policy_router import artifact_metadata, save_artifact, train_learned_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned offline policy router from a completed grid CSV.")
    parser.add_argument("csv", help="Training grid CSV containing candidate budget traces.")
    parser.add_argument("--out", default="results/models/learned_policy.joblib", help="Path to write the trained router artifact.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.02)
    args = parser.parse_args()

    artifact = train_learned_policy(args.csv, UtilityWeights(alpha=args.alpha, beta=args.beta, gamma=args.gamma))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(artifact, str(out_path))
    print(f"Wrote learned policy router to {out_path}")
    print(artifact_metadata(artifact))


if __name__ == "__main__":
    main()
