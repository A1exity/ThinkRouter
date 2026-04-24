from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from thinkrouter.official_protocol import OFFICIAL_PROTOCOL

SEMANTIC_EMBEDDING_DIM = int(os.getenv("THINKROUTER_SEMANTIC_DIM", "16"))
SEMANTIC_FEATURE_COLUMNS = tuple(f"semantic_embedding_{index:02d}" for index in range(SEMANTIC_EMBEDDING_DIM))
DEFAULT_SENTENCE_MODEL = os.getenv("THINKROUTER_SENTENCE_MODEL", OFFICIAL_PROTOCOL.semantic_model_name)

_FALLBACK_ANCHORS = [
    "Solve the arithmetic word problem and report the final number.",
    "Find the final boxed expression in the algebra problem.",
    "Write a Python function that passes the provided tests.",
    "Reason about numbers, units, and intermediate steps before answering.",
    "Read the code task and implement the requested function.",
    "Compute the result of the equation and simplify the answer.",
    "Return the correct output for the described algorithmic problem.",
    "Evaluate the geometry, algebra, or arithmetic expression carefully.",
    "Generate valid Python code with the required entry point and assertions.",
    "Answer the math story problem with one normalized final number.",
    "Determine whether the prompt is an easy, medium, or hard reasoning task.",
    "Choose the most cost-effective model and budget for this reasoning query.",
    "Explain the route confidence and fallback decision for the selected answer.",
    "Extract the semantic structure of a mathematical reasoning problem.",
    "Represent the query meaning with a compact semantic vector.",
    "Estimate similarity between this query and previously solved examples.",
]


class EmbeddingFeatureExtractor:
    name = "embedding"

    def __init__(self, dim: int = SEMANTIC_EMBEDDING_DIM) -> None:
        self.dim = dim
        self.columns = tuple(f"semantic_embedding_{index:02d}" for index in range(dim))

    def extract(self, query: str, task_type: str = "custom") -> dict[str, float]:
        encoder = _load_encoder(self.dim)
        vector = encoder.encode(query, task_type)
        return {column: float(value) for column, value in zip(self.columns, vector)}


class _SentenceTransformerEncoder:
    def __init__(self, dim: int) -> None:
        from sentence_transformers import SentenceTransformer

        self.dim = dim
        self.model = SentenceTransformer(DEFAULT_SENTENCE_MODEL)

    def encode(self, query: str, task_type: str = "custom") -> np.ndarray:
        text = f"[{task_type}] {query}"
        vector = np.asarray(self.model.encode(text, normalize_embeddings=True), dtype=float)
        if vector.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=float)
            padded[: vector.shape[0]] = vector
            return padded
        return vector[: self.dim]


class _LexicalSemanticFallback:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=256, stop_words="english")
        matrix = self.vectorizer.fit_transform(_FALLBACK_ANCHORS)
        n_components = max(1, min(dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self.reducer = TruncatedSVD(n_components=n_components, random_state=0)
        self.reducer.fit(matrix)

    def encode(self, query: str, task_type: str = "custom") -> np.ndarray:
        text = f"{task_type} {query}"
        matrix = self.vectorizer.transform([text])
        reduced = np.asarray(self.reducer.transform(matrix)[0], dtype=float)
        if reduced.shape[0] < self.dim:
            padded = np.zeros(self.dim, dtype=float)
            padded[: reduced.shape[0]] = reduced
            return padded
        return reduced[: self.dim]


@lru_cache(maxsize=1)
def _load_encoder(dim: int):
    try:
        return _SentenceTransformerEncoder(dim)
    except Exception:
        return _LexicalSemanticFallback(dim)
