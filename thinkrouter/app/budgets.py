from __future__ import annotations

from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field

from thinkrouter.official_protocol import OFFICIAL_PROTOCOL


class BudgetLevel(IntEnum):
    NO_THINK = 0
    BRIEF = 256
    MEDIUM = 1024


OFFICIAL_BUDGET_LEVELS: tuple[int, ...] = tuple(OFFICIAL_PROTOCOL.budgets)
BUDGET_LEVELS: tuple[int, ...] = OFFICIAL_BUDGET_LEVELS


BUDGET_PROMPTS: dict[int, str] = {
    0: "Answer directly. Do not include hidden reasoning. Keep the response concise.",
    256: "Use brief reasoning, then provide the final answer clearly.",
    1024: "Use a moderate amount of reasoning and check the answer before finalizing.",
}

EFFORT_BY_BUDGET: dict[int, str] = {
    0: "minimal",
    256: "brief",
    1024: "medium",
}

MAX_OUTPUT_TOKENS_BY_BUDGET: dict[int, int] = {
    0: 256,
    256: 600,
    1024: 1200,
}


class BudgetConfig(BaseModel):
    budget_id: str
    effort_level: str
    max_output_tokens: int
    provider_controls: dict[str, Any] = Field(default_factory=dict)
    prompt_template_version: str = "v2"
    legacy_budget: int


def budget_to_dict(config: BudgetConfig) -> dict[str, Any]:
    if hasattr(config, "model_dump"):
        return config.model_dump()  # type: ignore[attr-defined]
    return config.dict()


def validate_budget(budget: int) -> int:
    if budget not in BUDGET_LEVELS:
        raise ValueError(f"Unsupported budget {budget}. Expected one of {BUDGET_LEVELS}.")
    return budget


def compile_budget_config(budget: int | BudgetConfig) -> BudgetConfig:
    if isinstance(budget, BudgetConfig):
        validate_budget(budget.legacy_budget)
        return budget
    validated = validate_budget(int(budget))
    effort = EFFORT_BY_BUDGET[validated]
    provider_controls = {"reasoning_effort": effort}
    if validated == 0:
        provider_controls["reasoning_mode"] = "minimal"
    return BudgetConfig(
        budget_id=f"budget-{validated}",
        effort_level=effort,
        max_output_tokens=MAX_OUTPUT_TOKENS_BY_BUDGET[validated],
        provider_controls=provider_controls,
        legacy_budget=validated,
    )


def budget_instruction(budget: int | BudgetConfig) -> str:
    config = compile_budget_config(budget)
    return BUDGET_PROMPTS[config.legacy_budget]
