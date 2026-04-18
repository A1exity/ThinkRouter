from __future__ import annotations

import argparse
from pathlib import Path

from thinkrouter.experiments.learned_policy_router import artifact_metadata, calibrate_policy_artifact, load_artifact, save_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a learned policy artifact on a dev grid.")
    parser.add_argument("csv", help="Calibration grid CSV, typically a dev split.")
    parser.add_argument("--model", required=True, help="Path to an artifact produced by train_learned_policy.")
    parser.add_argument("--out", required=True, help="Path to write the calibrated artifact.")
    parser.add_argument("--summary-out", default=None, help="Optional CSV with calibration candidate policy scores.")
    args = parser.parse_args()

    artifact = load_artifact(args.model)
    calibrated, summary = calibrate_policy_artifact(artifact, args.csv)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_artifact(calibrated, str(out_path))
    print(f"Wrote calibrated learned policy router to {out_path}")
    print(artifact_metadata(calibrated))

    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False))
        print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
