from __future__ import annotations

import re
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any

from thinkrouter.app.schemas import EvalResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, output_text: str, expected_answer: str | None, metadata: dict[str, Any] | None = None) -> EvalResult:
        raise NotImplementedError


class GSM8KEvaluator(BaseEvaluator):
    def evaluate(self, output_text: str, expected_answer: str | None, metadata: dict[str, Any] | None = None) -> EvalResult:
        extracted = extract_numeric_answer(output_text)
        expected = normalize_numeric_answer(expected_answer)
        is_correct = bool(extracted is not None and expected is not None and extracted == expected)
        error_type = None
        if extracted is None:
            error_type = "parse_error"
        elif not is_correct:
            error_type = "wrong_answer"
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            parsed_output=extracted,
            extracted_answer=extracted,
            expected_answer=expected,
            error_type=error_type,
            judge_metadata={"task_type": "gsm8k", "comparison": "numeric_exact_match"},
        )


class ExactMatchEvaluator(BaseEvaluator):
    def evaluate(self, output_text: str, expected_answer: str | None, metadata: dict[str, Any] | None = None) -> EvalResult:
        extracted = extract_final_answer(output_text)
        expected = normalize_text(expected_answer)
        is_correct = bool(extracted and expected and normalize_text(extracted) == expected)
        error_type = None
        if not extracted:
            error_type = "empty_output"
        elif not is_correct:
            error_type = "wrong_answer"
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            parsed_output=extracted,
            extracted_answer=extracted,
            expected_answer=expected,
            error_type=error_type,
            judge_metadata={"task_type": "exact_match", "comparison": "normalized_text_exact_match"},
        )


class MATHEvaluator(BaseEvaluator):
    def evaluate(self, output_text: str, expected_answer: str | None, metadata: dict[str, Any] | None = None) -> EvalResult:
        extracted = normalize_math_answer(extract_math_output_answer(output_text))
        expected = normalize_math_answer(expected_answer)
        is_correct = bool(extracted and expected and math_answers_equal(extracted, expected))
        error_type = None
        if extracted is None:
            error_type = "parse_error"
        elif not is_correct:
            error_type = "wrong_answer"
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            parsed_output=extracted,
            extracted_answer=extracted,
            expected_answer=expected,
            error_type=error_type,
            judge_metadata={"task_type": "math", "comparison": "normalized_math_match"},
        )


class CodeEvaluator(BaseEvaluator):
    def evaluate(self, output_text: str, expected_answer: str | None, metadata: dict[str, Any] | None = None) -> EvalResult:
        metadata = metadata or {}
        extracted = extract_code_block(output_text)
        test_code = str(metadata.get("test_code") or "")
        entry_point = str(metadata.get("entry_point") or "")
        if not extracted:
            return EvalResult(
                score=0.0,
                is_correct=False,
                parsed_output=None,
                extracted_answer=None,
                expected_answer=expected_answer,
                error_type="parse_error",
                judge_metadata={"task_type": "humaneval", "entry_point": entry_point},
            )
        if not test_code:
            return EvalResult(
                score=0.0,
                is_correct=False,
                parsed_output=extracted,
                extracted_answer=extracted,
                expected_answer=expected_answer,
                error_type="execution_error",
                judge_metadata={"task_type": "humaneval", "entry_point": entry_point, "reason": "missing_test_code"},
            )
        namespace: dict[str, Any] = {}
        try:
            exec(extracted, namespace, namespace)
            exec(test_code, namespace, namespace)
        except AssertionError:
            return EvalResult(
                score=0.0,
                is_correct=False,
                parsed_output=extracted,
                extracted_answer=extracted,
                expected_answer=expected_answer,
                error_type="wrong_answer",
                judge_metadata={"task_type": "humaneval", "entry_point": entry_point, "tests_passed": False},
            )
        except Exception as exc:
            return EvalResult(
                score=0.0,
                is_correct=False,
                parsed_output=extracted,
                extracted_answer=extracted,
                expected_answer=expected_answer,
                error_type="execution_error",
                judge_metadata={"task_type": "humaneval", "entry_point": entry_point, "exception": type(exc).__name__},
            )
        return EvalResult(
            score=1.0,
            is_correct=True,
            parsed_output=extracted,
            extracted_answer=extracted,
            expected_answer=expected_answer,
            error_type=None,
            judge_metadata={"task_type": "humaneval", "entry_point": entry_point, "tests_passed": True},
        )


def get_evaluator(task_type: str) -> BaseEvaluator:
    if task_type == "gsm8k":
        return GSM8KEvaluator()
    if task_type == "math":
        return MATHEvaluator()
    if task_type == "humaneval":
        return CodeEvaluator()
    return ExactMatchEvaluator()


def extract_code_block(text: str) -> str | None:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced[-1].strip()
        return candidate or None
    function_match = re.search(r"(def\s+[a-zA-Z_]\w*\s*\(.*)", text, flags=re.DOTALL)
    if function_match:
        candidate = function_match.group(1).strip()
        return candidate or None
    return None


def extract_final_answer(text: str) -> str | None:
    marker_answer = extract_marked_answer(text)
    if marker_answer:
        return marker_answer
    return text.strip() if text.strip() else None


def extract_marked_answer(text: str) -> str | None:
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed
    patterns = [
        r"####\s*([^\n]+)",
        r"(?:final\s+answer|answer)\s*\*{0,2}\s*[:：]\s*\*{0,2}\s*([^\n]+)",
        r"(?:final\s+answer|answer)\s*\*{0,2}\s*[:：]\s*\*{0,2}\s*\n+\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            answer = _strip_markdown_answer(match.group(1))
            if answer:
                return answer
    return None


def extract_last_boxed(text: str) -> str | None:
    starts = [match.end() for match in re.finditer(r"\\boxed\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        chars: list[str] = []
        index = start
        while index < len(text):
            char = text[index]
            if char == "{":
                depth += 1
                chars.append(char)
            elif char == "}":
                depth -= 1
                if depth == 0:
                    value = "".join(chars).strip()
                    return value or None
                chars.append(char)
            else:
                chars.append(char)
            index += 1
    return None


def _strip_markdown_answer(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^[-*\s]+", "", value)
    value = value.replace("**", "").replace("__", "")
    return value.strip()


def extract_numeric_answer(text: str) -> str | None:
    marked = extract_marked_answer(text)
    if marked:
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", marked)
        if numbers:
            return normalize_numeric_answer(numbers[0])
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    return normalize_numeric_answer(numbers[-1])


def extract_math_output_answer(text: str) -> str | None:
    marked = extract_marked_answer(text)
    if marked:
        return marked
    for line in reversed([line.strip().strip("$") for line in text.splitlines() if line.strip()]):
        equation_match = re.search(r"(?:^|[\s,.;:])(?:[a-z]|answer|value)\s*=\s*(.+)$", line, flags=re.IGNORECASE)
        if equation_match:
            return equation_match.group(1).strip().strip("$.,;:")
    frac_matches = re.findall(r"\\[dt]?frac\s*\{[^{}]+\}\s*\{[^{}]+\}", text)
    if frac_matches:
        return frac_matches[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    stripped = text.strip()
    return stripped if stripped else None


def normalize_numeric_answer(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value)
    if not match:
        return None
    raw = match.group(0).replace(",", "")
    if "." in raw:
        normalized = raw.rstrip("0").rstrip(".")
        return normalized or "0"
    return raw


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.strip().lower().split())


def normalize_math_answer(value: str | None) -> str | None:
    if value is None:
        return None
    text = extract_last_boxed(value) or value
    text = text.strip()
    if not text:
        return None
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = re.sub(r"\s+", "", text)
    text = strip_outer_braces(text)
    if "=" in text:
        text = text.split("=")[-1]
        text = strip_outer_braces(text)
    if text.endswith("\\%"):
        text = text[:-2]
    elif text.endswith("%"):
        text = text[:-1]
    return text.lower() or None


def math_answers_equal(left: str, right: str) -> bool:
    if left == right:
        return True
    left_value = parse_simple_math_number(left)
    right_value = parse_simple_math_number(right)
    return left_value is not None and right_value is not None and left_value == right_value


def parse_simple_math_number(value: str | None) -> Fraction | None:
    if value is None:
        return None
    text = normalize_numeric_fragments(value)
    frac = re.fullmatch(r"([-+]?)\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", text)
    if frac:
        sign = -1 if frac.group(1) == "-" else 1
        denominator = int(frac.group(3))
        if denominator == 0:
            return None
        return sign * Fraction(int(frac.group(2)), denominator)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return Fraction(text)
    return None


def normalize_numeric_fragments(value: str) -> str:
    return value.replace(",", "").strip()


def strip_outer_braces(value: str) -> str:
    text = value
    while text.startswith("{") and text.endswith("}") and _outer_braces_wrap_text(text):
        text = text[1:-1]
    return text


def _outer_braces_wrap_text(value: str) -> bool:
    depth = 0
    for index, char in enumerate(value):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return False
    return depth == 0
