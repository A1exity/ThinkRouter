from __future__ import annotations

import time

from thinkrouter.adapters.base import BaseModelAdapter
from thinkrouter.app.budgets import budget_to_dict, validate_budget
from thinkrouter.app.schemas import ModelRequest, ModelResponse


class MockAdapter(BaseModelAdapter):
    def generate(self, request: ModelRequest) -> ModelResponse:
        start = time.perf_counter()
        budget_config = request.resolved_budget_config
        validate_budget(budget_config.legacy_budget)
        parsed_output = None
        if request.task_type == "humaneval" and request.metadata.get("canonical_solution"):
            parsed_output = str(request.metadata["canonical_solution"])
            output = f"```python\n{parsed_output}\n```"
        else:
            answer = request.metadata.get("expected_answer")
            if not answer:
                answer = _simple_math_answer(request.query) or "I do not know."
            parsed_output = str(answer)
            output = f"Mock response for {request.task_type}. Final answer: {answer}"
        prompt_tokens = max(1, len(request.query.split()))
        completion_tokens = max(1, len(output.split()))
        return ModelResponse(
            output_text=output,
            parsed_output=parsed_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_s=time.perf_counter() - start,
            finish_reason="stop",
            provider_meta={
                "backend": "mock",
                "provider": self.config.provider,
                "tier": self.config.tier,
                "budget_id": budget_config.budget_id,
            },
            raw_response={"backend": "mock", "budget": budget_to_dict(budget_config)},
        )


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
