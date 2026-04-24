from __future__ import annotations

import os
from pathlib import Path

from thinkrouter.app.schemas import ModelRequest, ModelResponse, model_copy_update
from thinkrouter.runtime.cache import RequestCache


def default_request_cache() -> RequestCache | None:
    if os.getenv("THINKROUTER_DISABLE_CACHE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    cache_path = Path(os.getenv("THINKROUTER_CACHE_PATH", "results/runtime/request_cache.sqlite"))
    return RequestCache(cache_path)


def generate_with_runtime(adapter, request: ModelRequest, cache: RequestCache | None = None) -> tuple[ModelResponse | None, dict[str, object]]:
    cache = cache or default_request_cache()
    if cache is not None:
        cached = cache.get(request)
        if cached is not None:
            return cached, {"cache_hit": True, "error_type": None, "error_message": None}
    try:
        response = adapter.generate(request)
    except Exception as exc:
        return None, {"cache_hit": False, "error_type": exc.__class__.__name__, "error_message": str(exc)}
    if cache is not None:
        cache.set(request, response)
    response = model_copy_update(
        response,
        {
            "provider_meta": {
                **dict(response.provider_meta),
                "cache_hit": False,
            }
        },
    )
    return response, {"cache_hit": False, "error_type": None, "error_message": None}
