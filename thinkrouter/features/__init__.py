from thinkrouter.features.base import FeaturePipeline
from thinkrouter.features.registry import default_feature_pipeline

DEFAULT_FEATURE_PIPELINE = default_feature_pipeline()


def extract_query_features(query: str, task_type: str = "custom") -> dict[str, float | str]:
    return DEFAULT_FEATURE_PIPELINE.extract(query, task_type)  # type: ignore[return-value]


def make_feature_frame(rows: list[dict[str, object]]):
    return DEFAULT_FEATURE_PIPELINE.make_frame(rows)


__all__ = [
    "FeaturePipeline",
    "DEFAULT_FEATURE_PIPELINE",
    "default_feature_pipeline",
    "extract_query_features",
    "make_feature_frame",
]
