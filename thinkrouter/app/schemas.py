from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

TaskType = Literal["gsm8k", "math", "humaneval", "custom"]
DifficultyLabel = Literal["easy", "medium", "hard"]


class ModelRequest(BaseModel):
    query: str
    task_type: TaskType = "custom"
    model_id: str
    budget: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    output_text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    raw_response: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    score: float
    is_correct: bool
    extracted_answer: str | None = None
    expected_answer: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    model_id: str
    budget: int
    difficulty: DifficultyLabel
    estimated_accuracy: float
    estimated_cost: float
    estimated_latency: float
    explanation: str


class TraceRecord(BaseModel):
    id: int | None = None
    query: str
    task_type: TaskType
    selected_model: str
    selected_budget: int
    output_text: str
    score: float
    is_correct: bool
    expected_answer: str | None = None
    extracted_answer: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    query: str
    task_type: TaskType = "custom"
    expected_answer: str | None = None
    model_id: str = "mock-cheap"
    budget: int = 0
    use_router: bool = False


class RunResponse(BaseModel):
    route: RouteDecision | None = None
    model_response: ModelResponse
    evaluation: EvalResult
    trace: TraceRecord


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[attr-defined]
    return model.dict()


def model_copy_update(model: BaseModel, update: dict[str, Any]) -> BaseModel:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)  # type: ignore[attr-defined]
    return model.copy(update=update)
