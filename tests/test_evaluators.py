from __future__ import annotations

from thinkrouter.app.evaluators import GSM8KEvaluator, extract_numeric_answer, normalize_numeric_answer


def test_extract_numeric_answer_prefers_final_answer() -> None:
    assert extract_numeric_answer("We compute 1 + 1 = 2. Final answer: 42") == "42"


def test_extract_numeric_answer_handles_markdown_answer_heading() -> None:
    text = "**Answer:**\nIt takes **160 minutes** (or 2 hours and 40 minutes) to download the file."

    assert extract_numeric_answer(text) == "160"


def test_extract_numeric_answer_uses_first_number_after_answer_marker() -> None:
    text = "Answer: John is **45 miles** from home at the end of those 4 hours."

    assert extract_numeric_answer(text) == "45"


def test_extract_numeric_answer_falls_back_to_last_number_without_marker() -> None:
    assert extract_numeric_answer("We compute 1 + 1 = 2, so 2") == "2"


def test_normalize_numeric_answer_removes_commas_and_trailing_zeroes() -> None:
    assert normalize_numeric_answer("1,200.00") == "1200"


def test_gsm8k_evaluator_exact_match() -> None:
    result = GSM8KEvaluator().evaluate("Final answer: 19", "19")
    assert result.is_correct is True
    assert result.score == 1.0