from __future__ import annotations

from thinkrouter.app.evaluators import GSM8KEvaluator, extract_numeric_answer, normalize_numeric_answer


def test_extract_numeric_answer_prefers_final_answer() -> None:
    assert extract_numeric_answer("We compute 1 + 1 = 2. Final answer: 42") == "42"


def test_normalize_numeric_answer_removes_commas_and_trailing_zeroes() -> None:
    assert normalize_numeric_answer("1,200.00") == "1200"


def test_gsm8k_evaluator_exact_match() -> None:
    result = GSM8KEvaluator().evaluate("Final answer: 19", "19")
    assert result.is_correct is True
    assert result.score == 1.0
