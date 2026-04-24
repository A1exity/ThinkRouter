from __future__ import annotations

import hashlib


class EmbeddingFeatureExtractor:
    name = "embedding"

    def extract(self, query: str, task_type: str = "custom") -> dict[str, float]:
        tokens = [token.strip(".,:;!?()[]{}").lower() for token in query.split() if token.strip()]
        if not tokens:
            return {
                "embedding_hash_mean": 0.0,
                "embedding_hash_std": 0.0,
                "semantic_bucket": 0.0,
                "semantic_diversity": 0.0,
            }
        values = [self._token_value(token) for token in tokens]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        bucket = float(int(mean * 10) % 17)
        diversity = float(len(set(tokens)) / max(len(tokens), 1))
        return {
            "embedding_hash_mean": float(mean),
            "embedding_hash_std": float(variance**0.5),
            "semantic_bucket": bucket,
            "semantic_diversity": diversity,
        }

    def _token_value(self, token: str) -> float:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF
