from __future__ import annotations

from thinkrouter.features.base import FeaturePipeline
from thinkrouter.features.cheap_probe import CheapProbeFeatureExtractor
from thinkrouter.features.embedding import EmbeddingFeatureExtractor
from thinkrouter.features.surface import SurfaceFeatureExtractor


def default_feature_pipeline() -> FeaturePipeline:
    return FeaturePipeline(
        extractors=[
            SurfaceFeatureExtractor(),
            EmbeddingFeatureExtractor(),
            CheapProbeFeatureExtractor(),
        ]
    )
