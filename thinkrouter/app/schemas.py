from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from thinkrouter.app.budgets import BudgetConfig, compile_budget_config

TaskType = Literal["gsm8k", "math", "humaneval", "custom"]
DifficultyLabel = Literal["easy", "medium", "hard"]


class ModelRequest(BaseModel):
    query: str
    task_type: TaskType = "custom"
    model_id: str
    budget: int = 0
    budget_config: BudgetConfig | None = None
    candidate_set_signature: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def resolved_budget_config(self) -> BudgetConfig:
        return self.budget_config or compile_budget_config(self.budget)


class ModelResponse(BaseModel):
    output_text: str
    parsed_output: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    error_type: str | None = None
    provider_meta: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    score: float
    is_correct: bool
    parsed_output: str | None = None
    extracted_answer: str | None = None
    expected_answer: str | None = None
    error_type: str | None = None
    judge_metadata: dict[str, Any] = Field(default_factory=dict)
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
    schema_version: int = 2
    query_id: str | None = None
    benchmark: str | None = None
    query: str
    query_text: str | None = None
    task_type: TaskType
    selected_model: str
    selected_model_provider: str | None = None
    selected_model_tier: str | None = None
    selected_model_alias: str | None = None
    selected_budget: int
    selected_budget_id: str | None = None
    budget_config: dict[str, Any] = Field(default_factory=dict)
    output_text: str
    parsed_output: str | None = None
    score: float
    is_correct: bool
    expected_answer: str | None = None
    extracted_answer: str | None = None
    judge_metadata: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    error_type: str | None = None
    router_features: dict[str, Any] = Field(default_factory=dict)
    route_reason: str | None = None
    candidate_set_signature: str | None = None
    prompt_template_version: str | None = None
    provider_response_meta: dict[str, Any] = Field(default_factory=dict)
    route_confidence: float | None = None
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.query_text is None:
            self.query_text = self.query
        if self.query_id is None:
            self.query_id = str(self.metadata.get("sample_id") or self.metadata.get("query_id") or "")
        if self.selected_budget_id is None and self.budget_config:
            self.selected_budget_id = str(self.budget_config.get("budget_id") or "")
        if not self.budget_config:
            self.budget_config = compile_budget_config(self.selected_budget).model_dump()
        if self.prompt_template_version is None:
            self.prompt_template_version = str(self.budget_config.get("prompt_template_version", "v2"))
        if self.selected_model_provider is None:
            self.selected_model_provider = str(self.provider_response_meta.get("provider") or self.metadata.get("selected_model_provider") or "")
        if self.selected_model_tier is None:
            self.selected_model_tier = str(self.provider_response_meta.get("tier") or self.metadata.get("selected_model_tier") or "")
        if self.selected_model_alias is None:
            self.selected_model_alias = str(self.metadata.get("selected_model_alias") or "")


class RunRequest(BaseModel):
    query: str
    task_type: TaskType = "custom"
    expected_answer: str | None = None
    model_id: str = "mock-cheap"
    budget: int = 0
    budget_preset: str | None = None
    candidate_models: list[str] | None = None
    candidate_budgets: list[int] | None = None
    router_name: str | None = None
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
