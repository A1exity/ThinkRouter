from __future__ import annotations


class SurfaceFeatureExtractor:
    name = "surface"

    def extract(self, query: str, task_type: str = "custom") -> dict[str, float | str]:
        text = query.lower()
        words = query.split()
        char_count = len(query)
        word_count = len(words)
        digit_count = sum(ch.isdigit() for ch in query)
        math_symbol_count = sum(ch in "+-*/=^" for ch in query)
        punctuation_count = sum(ch in ",.;:!?()[]{}" for ch in query)
        code_marker_count = sum(marker in text for marker in ["def ", "class ", "return", "assert", "```", "import "])
        digit_density = digit_count / max(char_count, 1)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        return {
            "char_count": float(char_count),
            "word_count": float(word_count),
            "digit_count": float(digit_count),
            "digit_density": float(digit_density),
            "math_symbol_count": float(math_symbol_count),
            "punctuation_count": float(punctuation_count),
            "code_marker_count": float(code_marker_count),
            "avg_word_length": float(avg_word_length),
        }
