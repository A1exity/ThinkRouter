from __future__ import annotations

from typing import Any

import pandas as pd

from thinkrouter.experiments.analyze_failures import parse_metadata


def add_sample_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "sample_id" not in out.columns:
        metadata_values = out["metadata"] if "metadata" in out.columns else [None] * len(out)
        row_ids = out["id"] if "id" in out.columns else range(len(out))
        out["sample_id"] = [parse_metadata(value).get("sample_id", str(row_id)) for value, row_id in zip(metadata_values, row_ids)]
    return out


def summarize_selection(name: str, selected: pd.DataFrame) -> dict[str, Any]:
    if selected.empty:
        return {"policy": name, "accuracy": 0.0, "avg_cost": 0.0, "total_cost": 0.0, "avg_latency": 0.0, "p95_latency": 0.0, "n": 0}
    return {
        "policy": name,
        "accuracy": float(selected["is_correct"].mean()),
        "avg_cost": float(selected["cost_usd"].mean()),
        "total_cost": float(selected["cost_usd"].sum()),
        "avg_latency": float(selected["latency_s"].mean()),
        "p95_latency": float(selected["latency_s"].quantile(0.95)),
        "n": int(len(selected)),
    }
