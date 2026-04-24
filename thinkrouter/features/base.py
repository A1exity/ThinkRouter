from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class FeatureExtractor(Protocol):
    name: str

    def extract(self, query: str, task_type: str = "custom") -> dict[str, float | str]:
        ...


@dataclass
class FeaturePipeline:
    extractors: list[FeatureExtractor]

    def extract(self, query: str, task_type: str = "custom", extra: dict[str, object] | None = None) -> dict[str, object]:
        merged: dict[str, object] = {"query": query, "task_type": task_type}
        for extractor in self.extractors:
            merged.update(extractor.extract(query, task_type))
        if extra:
            for key, value in extra.items():
                if key not in merged:
                    merged[key] = value
        return merged

    def make_frame(self, rows: list[dict[str, object]]) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for row in rows:
            query = str(row.get("query", ""))
            task_type = str(row.get("task_type", "custom"))
            records.append(self.extract(query, task_type, extra=row))
        return pd.DataFrame(records)
