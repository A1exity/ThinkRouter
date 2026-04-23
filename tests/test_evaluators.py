from __future__ import annotations

from thinkrouter.app.evaluators import CodeEvaluator, GSM8KEvaluator, MATHEvaluator, extract_code_block, extract_last_boxed, extract_math_output_answer, math_answers_equal, normalize_math_answer, extract_numeric_answer, normalize_numeric_answer


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
    assert result.error_type is None


def test_math_evaluator_matches_boxed_answer() -> None:
    result = MATHEvaluator().evaluate(r"We get \boxed{\frac{1}{2}}.", r"\frac{1}{2}")

    assert result.is_correct is True
    assert result.extracted_answer == r"\frac{1}{2}"


def test_gsm8k_evaluator_reports_parse_error_when_no_numeric_answer() -> None:
    result = GSM8KEvaluator().evaluate("I cannot solve this.", "19")

    assert result.is_correct is False
    assert result.error_type == "parse_error"


def test_math_normalization_handles_latex_spacing_and_wrappers() -> None:
    assert extract_last_boxed(r"First \boxed{1}, then \boxed{-\frac{1}{2}}") == r"-\frac{1}{2}"
    assert normalize_math_answer(r"$ \left\{ x \right\} $") == "x"


def test_extract_math_output_answer_falls_back_to_final_expression() -> None:
    assert extract_math_output_answer(r"After solving, x=\frac{9}{7}") == r"\frac{9}{7}"
    assert extract_math_output_answer("Equation: $2+1=3$.\n$x = \\dfrac{9}{7}$") == r"\dfrac{9}{7}"
    assert extract_math_output_answer("The graph has 2 vertical asymptotes.") == "2"


def test_math_answers_equal_handles_simple_numeric_equivalence() -> None:
    assert math_answers_equal(r"\frac{11}{2}", "5.5")
    assert normalize_math_answer(r"\frac{8}{2}=4") == "4"
    assert normalize_math_answer(r"7\%") == "7"


def test_code_evaluator_executes_python_solution() -> None:
    result = CodeEvaluator().evaluate(
        "```python\ndef add(a, b):\n    return a + b\n```",
        "pass",
        {"entry_point": "add", "test_code": "assert add(1, 2) == 3\nassert add(-1, 4) == 3\n"},
    )

    assert result.is_correct is True
    assert result.error_type is None


def test_code_evaluator_reports_execution_error_for_invalid_code() -> None:
    result = CodeEvaluator().evaluate(
        "```python\ndef add(a, b)\n    return a + b\n```",
        "pass",
        {"entry_point": "add", "test_code": "assert add(1, 2) == 3\n"},
    )

    assert result.is_correct is False
    assert result.error_type == "execution_error"


def test_extract_code_block_prefers_fenced_python() -> None:
    assert extract_code_block("Text\n```python\ndef square(x):\n    return x * x\n```") == "def square(x):\n    return x * x"
