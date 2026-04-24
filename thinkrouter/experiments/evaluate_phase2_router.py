from __future__ import annotations

import argparse
from pathlib import Path

from thinkrouter.experiments.phase2_router_replay import replay_router


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
