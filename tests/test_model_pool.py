from __future__ import annotations

from thinkrouter.app.models import default_model_configs, resolve_model_name


def test_default_model_configs_supports_qwen_pool_aliases(monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_MODEL_POOL", "qwen-flash,qwen-plus,qwen-max")
    monkeypatch.setenv("THINKROUTER_QWEN_FLASH_MODEL", "qwen3-flash")
    monkeypatch.setenv("THINKROUTER_QWEN_PLUS_MODEL", "qwen3-plus")
    monkeypatch.setenv("THINKROUTER_QWEN_MAX_MODEL", "qwen3-max")

    configs = list(default_model_configs().values())

    assert [config.model_id for config in configs] == ["qwen3-flash", "qwen3-plus", "qwen3-max"]
    assert [config.tier for config in configs] == ["cheap", "mid", "strong"]
    assert all(config.provider == "qwen" for config in configs)


def test_default_model_configs_preserves_legacy_two_tier_defaults(monkeypatch) -> None:
    monkeypatch.delenv("THINKROUTER_MODEL_POOL", raising=False)
    monkeypatch.delenv("THINKROUTER_MID_MODEL", raising=False)
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")

    configs = list(default_model_configs().values())

    assert [config.model_id for config in configs] == ["mock-cheap", "mock-strong"]


def test_resolve_model_name_supports_qwen_alias_without_pool(monkeypatch) -> None:
    monkeypatch.setenv("THINKROUTER_QWEN_PLUS_MODEL", "qwen3-plus")

    config = resolve_model_name("qwen-plus")

    assert config.model_id == "qwen3-plus"
    assert config.tier == "mid"
    assert config.provider == "qwen"
