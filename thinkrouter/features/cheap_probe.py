from __future__ import annotations


class CheapProbeFeatureExtractor:
    name = "cheap_probe"

    def extract(self, query: str, task_type: str = "custom") -> dict[str, float]:
        lowered = query.lower()
        length = len(query.split())
        digit_count = sum(ch.isdigit() for ch in query)
        code_markers = sum(marker in lowered for marker in ["def ", "assert", "return", "import ", "```"])
        math_markers = sum(ch in "+-*/=^" for ch in query)
        difficulty_score = min(1.0, (length / 80.0) + (digit_count / 20.0) + (code_markers / 2.0) + (math_markers / 10.0))
        confidence = max(0.05, 1.0 - difficulty_score * 0.75)
        consistency = max(0.0, 1.0 - abs(digit_count - math_markers) / max(length, 1))
        return {
            "cheap_probe_difficulty": float(difficulty_score),
            "cheap_probe_confidence": float(confidence),
            "cheap_probe_consistency": float(consistency),
        }
