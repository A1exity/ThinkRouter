from __future__ import annotations

from enum import IntEnum


class BudgetLevel(IntEnum):
    NO_THINK = 0
    BRIEF = 256
    MEDIUM = 1024
    DEEP = 4096


BUDGET_LEVELS: tuple[int, ...] = tuple(int(level) for level in BudgetLevel)


BUDGET_PROMPTS: dict[int, str] = {
    0: "Answer directly. Do not include hidden reasoning. Keep the response concise.",
    256: "Use brief reasoning, then provide the final answer clearly.",
    1024: "Use a moderate amount of reasoning and check the answer before finalizing.",
    4096: "Think carefully and deeply. Explore edge cases, verify the result, then provide the final answer.",
}


def validate_budget(budget: int) -> int:
    if budget not in BUDGET_LEVELS:
        raise ValueError(f"Unsupported budget {budget}. Expected one of {BUDGET_LEVELS}.")
    return budget


def budget_instruction(budget: int) -> str:
    return BUDGET_PROMPTS[validate_budget(budget)]
