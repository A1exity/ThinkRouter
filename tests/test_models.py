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
    assert response.total_tokens > 0
