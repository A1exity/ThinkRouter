from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from thinkrouter.app.budgets import budget_instruction, budget_to_dict, validate_budget
from thinkrouter.app.schemas import ModelRequest, ModelResponse


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    backend: str = "mock"
    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    cost_per_1k_tokens: float = 0.0


class BaseModelAdapter(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    def estimate_cost(self, total_tokens: int) -> float:
        return total_tokens / 1000.0 * self.config.cost_per_1k_tokens


class MockAdapter(BaseModelAdapter):
    def generate(self, request: ModelRequest) -> ModelResponse:
        start = time.perf_counter()
        budget_config = request.resolved_budget_config
        validate_budget(budget_config.legacy_budget)
        answer = request.metadata.get("expected_answer")
        if not answer:
            answer = _simple_math_answer(request.query) or "I do not know."
        output = f"Mock response for {request.task_type}. Final answer: {answer}"
        prompt_tokens = max(1, len(request.query.split()))
        completion_tokens = max(1, len(output.split()))
        return ModelResponse(
            output_text=output,
            parsed_output=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_s=time.perf_counter() - start,
            finish_reason="stop",
            provider_meta={"backend": "mock", "budget_id": budget_config.budget_id},
            raw_response={"backend": "mock", "budget": budget_to_dict(budget_config)},
        )


class OpenAICompatibleAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        api_key = config.api_key or os.getenv("THINKROUTER_OPENAI_API_KEY")
        base_url = config.base_url or os.getenv("THINKROUTER_OPENAI_BASE_URL")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package or use mock models.") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = config.model_name or config.model_id

    def generate(self, request: ModelRequest) -> ModelResponse:
        start = time.perf_counter()
        budget_config = request.resolved_budget_config
        instruction = budget_instruction(budget_config)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": request.query},
            ],
            max_completion_tokens=budget_config.max_output_tokens,
            temperature=0,
        )
        latency_s = time.perf_counter() - start
        content = response.choices[0].message.content or ""
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else prompt_tokens + completion_tokens
        return ModelResponse(
            output_text=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=latency_s,
            finish_reason=response.choices[0].finish_reason,
            provider_meta={"id": response.id, "model": response.model, "budget_id": budget_config.budget_id},
            raw_response={"id": response.id, "model": response.model, "budget": budget_to_dict(budget_config)},
        )


def build_adapter(config: ModelConfig) -> BaseModelAdapter:
    if config.backend == "mock":
        return MockAdapter(config)
    if config.backend in {"openai", "openai-compatible", "vllm"}:
        return OpenAICompatibleAdapter(config)
    raise ValueError(f"Unsupported model backend: {config.backend}")


def default_model_configs() -> dict[str, ModelConfig]:
    cheap_model = os.getenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    strong_model = os.getenv("THINKROUTER_STRONG_MODEL", "mock-strong")
    cheap_backend = "mock" if cheap_model.startswith("mock") else "openai-compatible"
    strong_backend = "mock" if strong_model.startswith("mock") else "openai-compatible"
    return {
        cheap_model: ModelConfig(
            model_id=cheap_model,
            backend=cheap_backend,
            model_name=cheap_model,
            cost_per_1k_tokens=float(os.getenv("THINKROUTER_CHEAP_COST_PER_1K", "0.0")),
        ),
        strong_model: ModelConfig(
            model_id=strong_model,
            backend=strong_backend,
            model_name=strong_model,
            cost_per_1k_tokens=float(os.getenv("THINKROUTER_STRONG_COST_PER_1K", "0.0006")),
        ),
    }


def _simple_math_answer(query: str) -> str | None:
    import re

    expression = re.findall(r"[-+*/() 0-9]+", query)
    candidates = [item.strip() for item in expression if any(ch.isdigit() for ch in item)]
    if not candidates:
        return None
    expr = max(candidates, key=len)
    if not re.fullmatch(r"[-+*/() 0-9]+", expr):
        return None
    try:
        return str(int(eval(expr, {"__builtins__": {}}, {})))
    except Exception:
        return None
