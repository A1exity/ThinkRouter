from __future__ import annotations

import pytest

from thinkrouter.app.budgets import budget_instruction, validate_budget


def test_validate_budget_accepts_fixed_levels() -> None:
    assert validate_budget(0) == 0
    assert validate_budget(256) == 256
    assert "brief" in budget_instruction(256).lower()


def test_validate_budget_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        validate_budget(512)
