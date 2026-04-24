from __future__ import annotations

from dataclasses import dataclass

from thinkrouter.app.schemas import ModelRequest, ModelResponse
from thinkrouter.runtime import RequestCache, generate_with_runtime


@dataclass
class _CountingAdapter:
    calls: int = 0

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1
        return ModelResponse(output_text="ok", total_tokens=10, provider_meta={"provider": "test"})


@dataclass
class _FailingAdapter:
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise TimeoutError("boom")


def test_generate_with_runtime_uses_cache(tmp_path) -> None:
    cache = RequestCache(tmp_path / "cache.sqlite")
    adapter = _CountingAdapter()
    request = ModelRequest(query="hello", task_type="custom", model_id="mock")

    first, first_meta = generate_with_runtime(adapter, request, cache)
    second, second_meta = generate_with_runtime(adapter, request, cache)

    assert first is not None
    assert second is not None
    assert adapter.calls == 1
    assert first_meta["cache_hit"] is False
    assert second_meta["cache_hit"] is True


def test_generate_with_runtime_returns_error_metadata_on_failure(tmp_path) -> None:
    cache = RequestCache(tmp_path / "cache.sqlite")
    response, meta = generate_with_runtime(_FailingAdapter(), ModelRequest(query="x", task_type="custom", model_id="mock"), cache)

    assert response is None
    assert meta["error_type"] == "TimeoutError"
