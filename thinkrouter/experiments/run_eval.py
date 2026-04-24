from __future__ import annotations

import argparse
from pathlib import Path

from thinkrouter.experiments.eval_baselines import summarize_baselines
from thinkrouter.experiments.make_phase2_report import build_phase2_report
from thinkrouter.experiments.make_plots import make_pareto_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation entrypoint for baseline and Phase 2 summaries.")
    parser.add_argument("csv", help="Completed grid CSV path.")
    parser.add_argument("--out-prefix", required=True, help="Output prefix, for example results/eval/qwen35_pool_gsm8k_dev20")
    parser.add_argument(
        "--phase2-router",
        action="append",
        default=[],
        help="Optional Phase 2 router replay spec. Use NAME or NAME=artifact_path.",
    )
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    summary_path = Path(f"{args.out_prefix}_baseline_summary.csv")
    ranked_path = Path(f"{args.out_prefix}_ranked.csv")
    report_path = Path(f"{args.out_prefix}_report.md")
    figure_path = Path(f"{args.out_prefix}_pareto.png")

    summary = summarize_baselines(args.csv, phase2_routers=args.phase2_router)
    summary.to_csv(summary_path, index=False)
    make_pareto_plot(args.csv, str(figure_path), phase2_routers=args.phase2_router)
    build_phase2_report(str(summary_path), str(ranked_path), str(report_path))

    print(f"Wrote {summary_path}")
    print(f"Wrote {ranked_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
