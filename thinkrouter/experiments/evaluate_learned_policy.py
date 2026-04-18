from __future__ import annotations

import argparse
from pathlib import Path

from thinkrouter.experiments.learned_policy_router import evaluate_learned_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a learned policy router on an evaluation grid CSV.")
    parser.add_argument("csv", help="Evaluation grid CSV containing candidate budget traces.")
    parser.add_argument("--model", required=True, help="Path to a joblib artifact produced by train_learned_policy.")
    parser.add_argument("--out", default="results/tables/learned_policy_summary.csv", help="Summary CSV output path.")
    parser.add_argument("--selected-out", default=None, help="Optional CSV of selected rows after replay.")
    parser.add_argument("--unsafe", action="store_true", help="Use raw classifier predictions instead of the safe fallback policy.")
    args = parser.parse_args()

    summary, selected = evaluate_learned_policy(args.csv, args.model, safe=not args.unsafe)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {out_path}")

    if args.selected_out:
        selected_path = Path(args.selected_out)
        selected_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(selected_path, index=False)
        print(f"Wrote {selected_path}")


if __name__ == "__main__":
    main()
