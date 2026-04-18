from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.analyze_failures import analyze_failures, classify_failure, parse_metadata


def test_classify_failure_detects_answer_format_extraction_error() -> None:
    output = "Answer: It takes 160 minutes, or 2 hours and 40 minutes."

    assert classify_failure(output, "160", "40") == "answer_format_extraction_error"


def test_classify_failure_detects_wrong_answer() -> None:
    output = "Final answer: 12"

    assert classify_failure(output, "13", "12") == "wrong_answer"


def test_classify_math_failure_detects_unmarked_correct_answer() -> None:
    output = r"Solving gives x=\frac{9}{7}, so the requested value is \frac{9}{7}."

    assert classify_failure(output, r"\frac{9}{7}", output, task_type="math") == "answer_format_extraction_error"


def test_parse_metadata_handles_python_dict_string() -> None:
    metadata = parse_metadata("{'sample_id': 'gsm8k_dev_001', 'split': 'dev'}")

    assert metadata["sample_id"] == "gsm8k_dev_001"
    assert metadata["split"] == "dev"


def test_analyze_failures_includes_cross_budget_correctness(tmp_path) -> None:
    csv_path = tmp_path / "grid.csv"
    rows = [
        {
            "id": 1,
            "query": "q",
            "task_type": "gsm8k",
            "selected_model": "model",
            "selected_budget": 0,
            "output_text": "Final answer: 13",
            "score": 1.0,
            "is_correct": True,
            "expected_answer": "13",
            "extracted_answer": "13",
            "total_tokens": 10,
            "cost_usd": 0.1,
            "latency_s": 1.0,
            "metadata": "{'sample_id': 's1', 'split': 'dev'}",
        },
        {
            "id": 2,
            "query": "q",
            "task_type": "gsm8k",
            "selected_model": "model",
            "selected_budget": 1024,
            "output_text": "Final answer: 12",
            "score": 0.0,
            "is_correct": False,
            "expected_answer": "13",
            "extracted_answer": "12",
            "total_tokens": 20,
            "cost_usd": 0.2,
            "latency_s": 2.0,
            "metadata": "{'sample_id': 's1', 'split': 'dev'}",
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    failures = analyze_failures(str(csv_path))

    assert len(failures) == 1
    assert failures.iloc[0]["sample_id"] == "s1"
    assert failures.iloc[0]["correct_budgets_for_sample"] == "0"
    assert bool(failures.iloc[0]["any_other_budget_correct"])
