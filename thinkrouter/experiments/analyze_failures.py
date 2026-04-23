from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd

from thinkrouter.app.evaluators import normalize_math_answer, normalize_numeric_answer


def parse_metadata(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def numeric_values(text: str) -> list[str]:
    values = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    normalized = [normalize_numeric_answer(value) for value in values]
    return [value for value in normalized if value is not None]


def classify_failure(output_text: str, expected_answer: object, extracted_answer: object, task_type: str = "gsm8k") -> str:
    if task_type == "math":
        return classify_math_failure(output_text, expected_answer, extracted_answer)
    if task_type == "humaneval":
        return classify_code_failure(output_text, extracted_answer)
    expected = normalize_numeric_answer(None if pd.isna(expected_answer) else str(expected_answer))
    extracted = normalize_numeric_answer(None if pd.isna(extracted_answer) else str(extracted_answer))
    if extracted is None:
        return "no_numeric_answer"
    if expected is None:
        return "missing_expected_answer"
    if extracted == expected:
        return "not_failure"
    numbers = numeric_values(output_text)
    if expected in numbers:
        return "answer_format_extraction_error"
    return "wrong_answer"


def classify_math_failure(output_text: str, expected_answer: object, extracted_answer: object) -> str:
    expected = normalize_math_answer(None if pd.isna(expected_answer) else str(expected_answer))
    extracted = normalize_math_answer(None if pd.isna(extracted_answer) else str(extracted_answer))
    if expected is None:
        return "missing_expected_answer"
    if extracted is None:
        return "no_final_answer"
    if extracted == expected:
        return "stale_or_normalization_mismatch"
    output_normalized = normalize_math_answer(output_text) or ""
    if expected in output_normalized:
        return "answer_format_extraction_error"
    return "wrong_answer"


def classify_code_failure(output_text: str, extracted_answer: object) -> str:
    extracted = "" if pd.isna(extracted_answer) else str(extracted_answer)
    if not extracted.strip():
        return "parse_error"
    normalized = output_text.lower()
    if "traceback" in normalized or "syntaxerror" in normalized:
        return "execution_error"
    if "def " not in extracted:
        return "malformed_answer"
    return "wrong_answer"


def summarize_correct_budgets(df: pd.DataFrame) -> dict[str, list[int]]:
    correct: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        metadata = parse_metadata(row.get("metadata"))
        sample_id = str(metadata.get("sample_id", row.get("id", "unknown")))
        if bool(row.get("is_correct")):
            correct.setdefault(sample_id, []).append(int(row["selected_budget"]))
        else:
            correct.setdefault(sample_id, [])
    return {sample_id: sorted(set(budgets)) for sample_id, budgets in correct.items()}


def analyze_failures(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame()
    correct_budgets = summarize_correct_budgets(df)
    failure_rows: list[dict[str, Any]] = []
    for _, row in df[~df["is_correct"].astype(bool)].iterrows():
        metadata = parse_metadata(row.get("metadata"))
        sample_id = str(metadata.get("sample_id", row.get("id", "unknown")))
        budgets = correct_budgets.get(sample_id, [])
        output_text = str(row.get("output_text", ""))
        failure_rows.append(
            {
                "sample_id": sample_id,
                "split": metadata.get("split"),
                "task_type": row.get("task_type"),
                "selected_model": row.get("selected_model"),
                "selected_budget": int(row.get("selected_budget")),
                "expected_answer": row.get("expected_answer"),
                "extracted_answer": row.get("extracted_answer"),
                "error_type": classify_failure(output_text, row.get("expected_answer"), row.get("extracted_answer"), str(row.get("task_type", "gsm8k"))),
                "correct_budgets_for_sample": ",".join(str(budget) for budget in budgets),
                "any_other_budget_correct": any(budget != int(row.get("selected_budget")) for budget in budgets),
                "total_tokens": int(row.get("total_tokens", 0)),
                "cost_usd": float(row.get("cost_usd", 0.0)),
                "latency_s": float(row.get("latency_s", 0.0)),
                "query": row.get("query"),
                "output_text": output_text,
                "output_preview": output_text[:240].replace("\n", " "),
            }
        )
    columns = [
        "sample_id",
        "split",
        "task_type",
        "selected_model",
        "selected_budget",
        "expected_answer",
        "extracted_answer",
        "error_type",
        "correct_budgets_for_sample",
        "any_other_budget_correct",
        "total_tokens",
        "cost_usd",
        "latency_s",
        "query",
        "output_preview",
        "output_text",
    ]
    return pd.DataFrame(failure_rows, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze incorrect ThinkRouter grid traces.")
    parser.add_argument("csv", help="Grid CSV path.")
    parser.add_argument("--out", default="results/tables/failures.csv", help="Failure analysis CSV output path.")
    args = parser.parse_args()

    failures = analyze_failures(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(out_path, index=False)
    print(failures[["sample_id", "selected_budget", "expected_answer", "extracted_answer", "error_type", "correct_budgets_for_sample"]].to_string(index=False) if not failures.empty else "No failures.")
    print(f"Wrote {len(failures)} failures to {out_path}")


if __name__ == "__main__":
    main()
