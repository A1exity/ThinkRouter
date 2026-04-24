from __future__ import annotations

import os
import time

from thinkrouter.adapters.base import BaseModelAdapter
from thinkrouter.app.budgets import budget_instruction, budget_to_dict
from thinkrouter.app.schemas import ModelRequest, ModelResponse


class OpenAICompatibleAdapter(BaseModelAdapter):
    def __init__(self, config) -> None:
        super().__init__(config)
        api_key = config.api_key or os.getenv("THINKROUTER_OPENAI_API_KEY")
        base_url = config.base_url or os.getenv("THINKROUTER_OPENAI_BASE_URL")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package or use mock models.") from exc
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = config.model_name or config.model_id
        self.max_retries = max(0, int(os.getenv("THINKROUTER_PROVIDER_MAX_RETRIES", "2")))
        self.retry_backoff_s = max(0.0, float(os.getenv("THINKROUTER_PROVIDER_RETRY_BACKOFF_S", "2.0")))

    def generate(self, request: ModelRequest) -> ModelResponse:
        budget_config = request.resolved_budget_config
        instruction = budget_instruction(budget_config)
        response = None
        latency_s = 0.0
        last_error = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
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
                break
            except Exception as exc:
                latency_s = time.perf_counter() - start
                last_error = exc
                if not _is_retryable_openai_error(exc) or attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_backoff_s * (attempt + 1))
        if response is None:
            assert last_error is not None
            raise last_error
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
            provider_meta={
                "id": response.id,
                "model": response.model,
                "provider": self.config.provider,
                "tier": self.config.tier,
                "budget_id": budget_config.budget_id,
            },
            raw_response={"id": response.id, "model": response.model, "budget": budget_to_dict(budget_config)},
        )


def _is_retryable_openai_error(exc: Exception) -> bool:
    name = exc.__class__.__name__
    return name in {"APITimeoutError", "APIConnectionError", "RateLimitError", "InternalServerError"}
