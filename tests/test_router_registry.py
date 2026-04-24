from __future__ import annotations

import pandas as pd

from thinkrouter.adapters.base import ModelConfig
from thinkrouter.routers import MLPFactorizedRouter, ThresholdRouter, UncertaintyAwareRouter, available_routers, train_factorized_router, train_logreg_joint_router
from thinkrouter.training import derive_factorized_examples, derive_joint_examples


def _models() -> list[ModelConfig]:
    return [
        ModelConfig(model_id="mock-cheap", backend="mock", provider="mock", tier="cheap"),
        ModelConfig(model_id="mock-mid", backend="mock", provider="mock", tier="mid"),
        ModelConfig(model_id="mock-strong", backend="mock", provider="mock", tier="strong"),
    ]


def _rows() -> list[dict[str, object]]:
    return [
        {
            "sample_id": "s1",
            "query": "Add 1 and 2.",
            "task_type": "gsm8k",
            "selected_model": "mock-cheap",
            "selected_budget": 0,
            "is_correct": 1,
            "cost_usd": 0.0001,
            "latency_s": 0.4,
        },
        {
            "sample_id": "s2",
            "query": "Solve x^2+5x+6=0.",
            "task_type": "math",
            "selected_model": "mock-mid",
            "selected_budget": 256,
            "is_correct": 1,
            "cost_usd": 0.0004,
            "latency_s": 0.8,
        },
        {
            "sample_id": "s3",
            "query": "Write a Python function reverse_string(s).",
            "task_type": "humaneval",
            "selected_model": "mock-strong",
            "selected_budget": 1024,
            "is_correct": 1,
            "cost_usd": 0.001,
            "latency_s": 1.6,
        },
    ]


def test_threshold_router_returns_confidence_and_name() -> None:
    decision = ThresholdRouter(_models()).route("Write a Python function factorial(n).", "humaneval")

    assert decision.router_name == "threshold"
    assert decision.route_confidence is not None


def test_factorized_router_uses_trained_artifact() -> None:
    artifact = train_factorized_router(derive_factorized_examples(pd.DataFrame(_rows())))
    decision = MLPFactorizedRouter(_models(), artifact=artifact).route("Solve x^2 + 5x + 6 = 0.", "math")

    assert decision.router_name == "mlp_factorized"
    assert decision.budget in {0, 256, 1024, 4096}


def test_uncertainty_router_falls_back_without_artifact() -> None:
    decision = UncertaintyAwareRouter(_models(), artifact=None).route("Add 1 and 2.", "gsm8k")

    assert decision.router_name == "uncertainty_aware"
    assert decision.fallback_triggered is True
    assert decision.fallback_reason == "missing_artifact"


def test_available_routers_lists_phase2_set() -> None:
    assert set(available_routers()) == {"threshold", "logreg_joint", "mlp_factorized", "uncertainty_aware"}


def test_logreg_joint_training_runs() -> None:
    artifact = train_logreg_joint_router(derive_joint_examples(pd.DataFrame(_rows())))
    assert artifact.labels
