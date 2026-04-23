from __future__ import annotations

import pytest

from thinkrouter.app.budgets import budget_instruction, compile_budget_config, validate_budget


def test_validate_budget_accepts_fixed_levels() -> None:
    assert validate_budget(0) == 0
    assert validate_budget(256) == 256
    assert "brief" in budget_instruction(256).lower()


def test_validate_budget_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        validate_budget(512)


def test_compile_budget_config_keeps_legacy_budget_and_structured_fields() -> None:
    config = compile_budget_config(1024)

    assert config.legacy_budget == 1024
    assert config.budget_id == "budget-1024"
    assert config.max_output_tokens >= 1000
    assert config.provider_controls["reasoning_effort"] == "medium"
