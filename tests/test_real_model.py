from __future__ import annotations

from thinkrouter.experiments.real_model import check_openai_compatible_config


def test_real_model_config_reports_missing_for_mock_defaults(monkeypatch) -> None:
    monkeypatch.delenv("THINKROUTER_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("THINKROUTER_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")

    check = check_openai_compatible_config()

    assert not check.ok
    assert "THINKROUTER_OPENAI_BASE_URL" in check.missing
    assert "THINKROUTER_OPENAI_API_KEY" in check.missing
    assert any("non-mock" in item for item in check.missing)


def test_real_model_config_accepts_non_mock_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("THINKROUTER_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "gpt-test")

    check = check_openai_compatible_config()

    assert check.ok
    assert check.model_id == "gpt-test"
    assert check.base_url == "https://example.test/v1"
    assert check.missing == []


def test_real_model_config_model_argument_overrides_env(monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("THINKROUTER_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")

    check = check_openai_compatible_config("provider-model")

    assert check.ok
    assert check.model_id == "provider-model"