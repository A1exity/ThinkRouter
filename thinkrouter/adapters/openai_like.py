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
            provider_meta={
                "id": response.id,
                "model": response.model,
                "provider": self.config.provider,
                "tier": self.config.tier,
                "budget_id": budget_config.budget_id,
            },
            raw_response={"id": response.id, "model": response.model, "budget": budget_to_dict(budget_config)},
        )
