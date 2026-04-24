from __future__ import annotations

import math
import random

import pandas as pd


def bootstrap_metric_ci(
    values: list[float] | pd.Series,
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    series = pd.Series(values, dtype="float64").dropna()
    if series.empty:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng = random.Random(seed)
    samples: list[float] = []
    raw = series.tolist()
    for _ in range(max(1, num_samples)):
        draw = [raw[rng.randrange(len(raw))] for _ in range(len(raw))]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    alpha = max(0.0, min(1.0, 1.0 - confidence))
    lower_index = int(math.floor((alpha / 2.0) * (len(samples) - 1)))
    upper_index = int(math.ceil((1.0 - alpha / 2.0) * (len(samples) - 1)))
    return {
        "mean": float(series.mean()),
        "lower": float(samples[lower_index]),
        "upper": float(samples[upper_index]),
    }
