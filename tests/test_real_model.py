from __future__ import annotations

from thinkrouter.experiments import real_model
from thinkrouter.experiments.real_model import check_openai_compatible_config


def test_real_model_config_reports_missing_for_mock_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("THINKROUTER_OPENAI_BASE_URL", "")
    monkeypatch.setenv("THINKROUTER_OPENAI_API_KEY", "")
    monkeypatch.setenv("THINKROUTER_MODEL_POOL", "")
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_MID_MODEL", "")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")

    check = check_openai_compatible_config()

    assert not check.ok
    assert "THINKROUTER_OPENAI_BASE_URL" in check.missing
    assert "THINKROUTER_OPENAI_API_KEY" in check.missing
    assert any("non-mock" in item for item in check.missing)


def test_real_model_config_accepts_non_mock_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("THINKROUTER_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("THINKROUTER_MODEL_POOL", "")
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_MID_MODEL", "")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "gpt-test")

    check = check_openai_compatible_config()

    assert check.ok
    assert check.model_id == "gpt-test"
    assert check.base_url == "https://example.test/v1"
    assert check.missing == []


def test_real_model_config_model_argument_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("THINKROUTER_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("THINKROUTER_MODEL_POOL", "")
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_MID_MODEL", "")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")

    check = check_openai_compatible_config("provider-model")

    assert check.ok
    assert check.model_id == "provider-model"


def test_real_model_config_uses_strongest_model_from_pool(monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("THINKROUTER_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("THINKROUTER_MODEL_POOL", "qwen-flash,qwen-plus,qwen-max")
    monkeypatch.setenv("THINKROUTER_QWEN_FLASH_MODEL", "qwen3-flash")
    monkeypatch.setenv("THINKROUTER_QWEN_PLUS_MODEL", "qwen3-plus")
    monkeypatch.setenv("THINKROUTER_QWEN_MAX_MODEL", "qwen3-max")

    check = check_openai_compatible_config()

    assert check.ok
    assert check.model_id == "qwen3-max"


def test_run_openai_compatible_smoke_uses_resolved_model_cost(monkeypatch) -> None:
    class FakeAdapter:
        def __init__(self, config):
            self.config = config

        def generate(self, request):
            from thinkrouter.app.schemas import ModelResponse

            return ModelResponse(output_text="ok", total_tokens=1000)

        def estimate_cost(self, total_tokens: int) -> float:
            return total_tokens / 1000.0 * self.config.cost_per_1k_tokens

    monkeypatch.setattr(real_model, "resolve_model_name", lambda model_id: real_model.ModelConfig(model_id=model_id, backend="openai-compatible", model_name=model_id, cost_per_1k_tokens=0.123))
    monkeypatch.setattr(real_model, "build_adapter", lambda config: FakeAdapter(config))

    _, estimated_cost = real_model.run_openai_compatible_smoke("qwen-test", "q")

    assert estimated_cost == 0.123
