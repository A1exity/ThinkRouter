from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from thinkrouter.app.models import ModelConfig, build_adapter, default_primary_model_id, resolve_model_name
from thinkrouter.app.schemas import ModelRequest, ModelResponse

PLACEHOLDER_API_KEYS = {"", "replace-me", "changeme", "your-api-key"}


@dataclass(frozen=True)
class EndpointCheck:
    ok: bool
    model_id: str
    base_url: str | None
    missing: list[str]
    backend: str


def check_openai_compatible_config(model_id: str | None = None) -> EndpointCheck:
    load_dotenv()
    selected_model = model_id or default_primary_model_id()
    base_url = os.getenv("THINKROUTER_OPENAI_BASE_URL")
    api_key = os.getenv("THINKROUTER_OPENAI_API_KEY", "")
    missing: list[str] = []
    if not selected_model or selected_model.startswith("mock"):
        missing.append("THINKROUTER_MODEL_POOL / THINKROUTER_STRONG_MODEL / --model must provide a non-mock model id")
    if not base_url:
        missing.append("THINKROUTER_OPENAI_BASE_URL")
    if api_key.strip().lower() in PLACEHOLDER_API_KEYS:
        missing.append("THINKROUTER_OPENAI_API_KEY")
    return EndpointCheck(
        ok=not missing,
        model_id=selected_model,
        base_url=base_url,
        missing=missing,
        backend="openai-compatible",
    )


def run_openai_compatible_smoke(
    model_id: str,
    query: str,
    task_type: str = "gsm8k",
    expected_answer: str | None = None,
    budget: int = 0,
    cost_per_1k_tokens: float | None = None,
) -> tuple[ModelResponse, float]:
    load_dotenv()
    resolved = resolve_model_name(model_id)
    config = ModelConfig(
        model_id=resolved.model_id,
        backend=resolved.backend,
        model_name=resolved.model_name,
        base_url=resolved.base_url,
        api_key=resolved.api_key,
        cost_per_1k_tokens=cost_per_1k_tokens if cost_per_1k_tokens is not None else resolved.cost_per_1k_tokens,
        provider=resolved.provider,
        tier=resolved.tier,
        alias=resolved.alias,
    )
    adapter = build_adapter(config)
    response = adapter.generate(
        ModelRequest(
            query=query,
            task_type=task_type,  # type: ignore[arg-type]
            model_id=config.model_id,
            budget=budget,
            metadata={"expected_answer": expected_answer},
        )
    )
    return response, adapter.estimate_cost(response.total_tokens)
