from __future__ import annotations

from thinkrouter.app.models import MockAdapter, ModelConfig
from thinkrouter.app.schemas import ModelRequest


def test_mock_adapter_returns_expected_answer() -> None:
    adapter = MockAdapter(ModelConfig(model_id="mock-cheap"))
    response = adapter.generate(
        ModelRequest(
            query="What is 2 + 2?",
            task_type="gsm8k",
            model_id="mock-cheap",
            budget=0,
            metadata={"expected_answer": "4"},
        )
    )
    assert "Final answer: 4" in response.output_text
    assert response.parsed_output == "4"
    assert response.provider_meta["budget_id"] == "budget-0"
    assert response.total_tokens > 0


def test_mock_adapter_returns_canonical_code_for_humaneval() -> None:
    adapter = MockAdapter(ModelConfig(model_id="mock-cheap"))
    response = adapter.generate(
        ModelRequest(
            query="Write add.",
            task_type="humaneval",
            model_id="mock-cheap",
            budget=0,
            metadata={"canonical_solution": "def add(a, b):\n    return a + b\n"},
        )
    )

    assert "def add(a, b)" in response.output_text
    assert response.parsed_output == "def add(a, b):\n    return a + b\n"
