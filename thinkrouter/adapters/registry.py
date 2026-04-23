from __future__ import annotations

import os

from thinkrouter.adapters.base import BaseModelAdapter, ModelConfig
from thinkrouter.adapters.mock import MockAdapter
from thinkrouter.adapters.openai_like import OpenAICompatibleAdapter

DEFAULT_MOCK_MODELS = ("mock-cheap", "mock-strong")
LEGACY_TIER_KEYS = ("cheap", "mid", "strong")
QWEN_ALIAS_META: dict[str, tuple[str, str, str, str]] = {
    "qwen-flash": ("THINKROUTER_QWEN_FLASH_MODEL", "THINKROUTER_QWEN_FLASH_COST_PER_1K", "cheap", "openai-compatible"),
    "qwen-plus": ("THINKROUTER_QWEN_PLUS_MODEL", "THINKROUTER_QWEN_PLUS_COST_PER_1K", "mid", "openai-compatible"),
    "qwen-max": ("THINKROUTER_QWEN_MAX_MODEL", "THINKROUTER_QWEN_MAX_COST_PER_1K", "strong", "openai-compatible"),
}
QWEN_DEFAULT_MODEL_IDS = {
    "qwen-flash": "qwen3.5-flash-2026-02-23",
    "qwen-plus": "qwen3.5-plus-2026-02-15",
    "qwen-max": "qwen3-max-2026-01-23",
}
QWEN_DEFAULT_COSTS = {
    "qwen-flash": "0.0003",
    "qwen-plus": "0.0012",
    "qwen-max": "0.0040",
}


def build_adapter(config: ModelConfig) -> BaseModelAdapter:
    if config.backend == "mock":
        return MockAdapter(config)
    if config.backend in {"openai", "openai-compatible", "vllm"}:
        return OpenAICompatibleAdapter(config)
    raise ValueError(f"Unsupported model backend: {config.backend}")


def default_model_configs() -> dict[str, ModelConfig]:
    pool = parse_model_pool()
    return {config.model_id: config for config in pool}


def default_primary_model_id() -> str:
    configs = list(default_model_configs().values())
    if not configs:
        return ""
    ranked = sorted(configs, key=_tier_rank)
    return ranked[-1].model_id


def parse_model_pool() -> list[ModelConfig]:
    configured_pool = os.getenv("THINKROUTER_MODEL_POOL")
    if configured_pool:
        names = [item.strip() for item in configured_pool.split(",") if item.strip()]
        if names:
            return [resolve_model_name(name) for name in names]

    legacy = _legacy_tier_configs()
    if legacy:
        return legacy
    return [resolve_model_name(name) for name in DEFAULT_MOCK_MODELS]


def resolve_model_name(name: str) -> ModelConfig:
    return _resolve_named_model(name)


def _legacy_tier_configs() -> list[ModelConfig]:
    tier_to_env = {
        "cheap": ("THINKROUTER_CHEAP_MODEL", "THINKROUTER_CHEAP_COST_PER_1K"),
        "mid": ("THINKROUTER_MID_MODEL", "THINKROUTER_MID_COST_PER_1K"),
        "strong": ("THINKROUTER_STRONG_MODEL", "THINKROUTER_STRONG_COST_PER_1K"),
    }
    defaults = {"cheap": "mock-cheap", "strong": "mock-strong"}
    configs: list[ModelConfig] = []
    for tier in LEGACY_TIER_KEYS:
        model_env, cost_env = tier_to_env[tier]
        model_id = os.getenv(model_env, defaults.get(tier, "")).strip()
        if not model_id:
            continue
        backend = "mock" if model_id.startswith("mock") else os.getenv("THINKROUTER_DEFAULT_BACKEND", "openai-compatible")
        configs.append(
            ModelConfig(
                model_id=model_id,
                backend=backend,
                model_name=model_id,
                base_url=os.getenv("THINKROUTER_OPENAI_BASE_URL") or None,
                api_key=os.getenv("THINKROUTER_OPENAI_API_KEY") or None,
                cost_per_1k_tokens=float(os.getenv(cost_env, "0.0" if tier != "strong" else "0.0006")),
                provider="mock" if backend == "mock" else "openai-compatible",
                tier=tier,
            )
        )
    return configs


def _resolve_named_model(name: str) -> ModelConfig:
    if name in QWEN_ALIAS_META:
        model_env, cost_env, tier, backend = QWEN_ALIAS_META[name]
        model_id = os.getenv(model_env, QWEN_DEFAULT_MODEL_IDS[name]).strip() or QWEN_DEFAULT_MODEL_IDS[name]
        effective_backend = "mock" if model_id.startswith("mock") else backend
        return ModelConfig(
            model_id=model_id,
            backend=effective_backend,
            model_name=model_id,
            base_url=os.getenv("THINKROUTER_OPENAI_BASE_URL") or None,
            api_key=os.getenv("THINKROUTER_OPENAI_API_KEY") or None,
            cost_per_1k_tokens=float(os.getenv(cost_env, QWEN_DEFAULT_COSTS[name])),
            provider="qwen",
            tier=tier,
            alias=name,
        )
    backend = "mock" if name.startswith("mock") else os.getenv("THINKROUTER_DEFAULT_BACKEND", "openai-compatible")
    inferred_tier = _infer_tier_from_name(name)
    default_cost = "0.0" if backend == "mock" else "0.0006"
    return ModelConfig(
        model_id=name,
        backend=backend,
        model_name=name,
        base_url=os.getenv("THINKROUTER_OPENAI_BASE_URL") or None,
        api_key=os.getenv("THINKROUTER_OPENAI_API_KEY") or None,
        cost_per_1k_tokens=float(default_cost),
        provider="mock" if backend == "mock" else "openai-compatible",
        tier=inferred_tier,
        alias=name,
    )


def _infer_tier_from_name(name: str) -> str | None:
    lowered = name.lower()
    if any(token in lowered for token in ("flash", "cheap", "mini")):
        return "cheap"
    if any(token in lowered for token in ("plus", "mid")):
        return "mid"
    if any(token in lowered for token in ("max", "strong", "pro")):
        return "strong"
    return None


def _tier_rank(config: ModelConfig) -> int:
    order = {"cheap": 0, "mid": 1, "strong": 2}
    return order.get(config.tier or "", -1)
