from __future__ import annotations

import os
from pathlib import Path

from thinkrouter.adapters.base import ModelConfig
from thinkrouter.official_protocol import OFFICIAL_PROTOCOL
from thinkrouter.routers.common import RouterWeights
from thinkrouter.routers.logreg_joint import LogRegJointRouter, load_logreg_joint_artifact
from thinkrouter.routers.mlp_factorized import MLPFactorizedRouter, load_factorized_artifact
from thinkrouter.routers.threshold import ThresholdRouter
from thinkrouter.routers.uncertainty_aware import UncertaintyAwareRouter


def build_router(router_name: str | None, models: list[ModelConfig], weights: RouterWeights | None = None):
    default_name = os.getenv("THINKROUTER_ROUTER", OFFICIAL_PROTOCOL.default_router)
    name = (router_name or default_name).strip().lower()
    weights = weights or RouterWeights()
    if name == "threshold":
        return ThresholdRouter(models, weights)
    if name == "logreg_joint":
        artifact_path = os.getenv("THINKROUTER_LOGREG_JOINT_MODEL_PATH")
        artifact = load_logreg_joint_artifact(artifact_path) if artifact_path and Path(artifact_path).exists() else None
        return LogRegJointRouter(models, artifact=artifact, weights=weights)
    if name == "mlp_factorized":
        artifact_path = os.getenv("THINKROUTER_FACTORIZED_ROUTER_MODEL_PATH")
        artifact = load_factorized_artifact(artifact_path) if artifact_path and Path(artifact_path).exists() else None
        return MLPFactorizedRouter(models, artifact=artifact, weights=weights)
    if name == "uncertainty_aware":
        artifact_path = os.getenv("THINKROUTER_FACTORIZED_ROUTER_MODEL_PATH")
        artifact = load_factorized_artifact(artifact_path) if artifact_path and Path(artifact_path).exists() else None
        threshold = float(os.getenv("THINKROUTER_UNCERTAINTY_THRESHOLD", "0.55"))
        return UncertaintyAwareRouter(models, artifact=artifact, weights=weights, confidence_threshold=threshold)
    raise ValueError(f"Unsupported router '{router_name}'.")


def available_routers() -> list[str]:
    return ["threshold", "logreg_joint", "mlp_factorized", "uncertainty_aware"]
