from __future__ import annotations

import re
from abc import ABC, abstractmethod

from thinkrouter.app.schemas import EvalResult


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, output_text: str, expected_answer: str | None) -> EvalResult:
        raise NotImplementedError


class GSM8KEvaluator(BaseEvaluator):
    def evaluate(self, output_text: str, expected_answer: str | None) -> EvalResult:
        extracted = extract_numeric_answer(output_text)
        expected = normalize_numeric_answer(expected_answer)
        is_correct = bool(extracted is not None and expected is not None and extracted == expected)
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            extracted_answer=extracted,
            expected_answer=expected,
        )


class ExactMatchEvaluator(BaseEvaluator):
    def evaluate(self, output_text: str, expected_answer: str | None) -> EvalResult:
        extracted = extract_final_answer(output_text)
        expected = normalize_text(expected_answer)
        is_correct = bool(extracted and expected and normalize_text(extracted) == expected)
        return EvalResult(
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            extracted_answer=extracted,
            expected_answer=expected,
        )


def get_evaluator(task_type: str) -> BaseEvaluator:
    if task_type == "gsm8k":
        return GSM8KEvaluator()
    return ExactMatchEvaluator()


def extract_final_answer(text: str) -> str | None:
    patterns = [
        r"####\s*([^\n]+)",
        r"final answer\s*[:?]\s*([^\n]+)",
        r"answer\s*[:?]\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return text.strip() if text.strip() else None


def extract_numeric_answer(text: str) -> str | None:
    final = extract_final_answer(text) or text
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", final)
    if not numbers:
        numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not numbers:
        return None
    return normalize_numeric_answer(numbers[-1])


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
