from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from thinkrouter.app.evaluators import get_evaluator


def regrade_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    required = {"task_type", "output_text", "expected_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    scores: list[float] = []
    is_correct: list[bool] = []
    extracted_answers: list[str | None] = []
    expected_answers: list[str | None] = []
    for row in out.itertuples(index=False):
        task_type = str(getattr(row, "task_type"))
        output_text = str(getattr(row, "output_text"))
        expected_answer = getattr(row, "expected_answer")
        expected = None if pd.isna(expected_answer) else str(expected_answer)
        result = get_evaluator(task_type).evaluate(output_text, expected)
        scores.append(result.score)
        is_correct.append(result.is_correct)
        extracted_answers.append(result.extracted_answer)
        expected_answers.append(result.expected_answer)
    out["score"] = scores
    out["is_correct"] = is_correct
    out["extracted_answer"] = extracted_answers
    out["expected_answer"] = expected_answers
    return out


def regrade_csv(csv_path: str) -> pd.DataFrame:
    return regrade_dataframe(pd.read_csv(csv_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-grade a ThinkRouter trace CSV with the current evaluator code.")
    parser.add_argument("csv", help="Input trace CSV path.")
    parser.add_argument("--out", required=True, help="Output regraded CSV path.")
    args = parser.parse_args()

    df = regrade_csv(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    accuracy = df["is_correct"].mean() if not df.empty else 0.0
    changed = 0
    original = pd.read_csv(args.csv)
    if "is_correct" in original.columns and len(original) == len(df):
        changed = int((original["is_correct"].astype(bool) != df["is_correct"].astype(bool)).sum())
    print(f"Wrote {len(df)} regraded traces to {out_path}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Changed correctness rows: {changed}")


if __name__ == "__main__":
    main()