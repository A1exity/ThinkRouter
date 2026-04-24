from __future__ import annotations

import pandas as pd

from thinkrouter.experiments.analyze_failures import analyze_failures


def build_failure_browser_frame(csv_path: str) -> pd.DataFrame:
    failures = analyze_failures(csv_path)
    if failures.empty:
        return failures
    out = failures.copy()
    out["confidence_band"] = out.get("route_confidence", pd.Series([pd.NA] * len(out)))
    if "route_confidence" in out.columns:
        confidence = pd.to_numeric(out["route_confidence"], errors="coerce")
        out["confidence_band"] = pd.cut(
            confidence,
            bins=[-0.001, 0.25, 0.5, 0.75, 1.0],
            labels=["very_low", "low", "medium", "high"],
        ).astype("string")
    return out
