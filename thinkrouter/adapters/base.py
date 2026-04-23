from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from thinkrouter.app.schemas import ModelRequest, ModelResponse


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    backend: str = "mock"
    model_name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    cost_per_1k_tokens: float = 0.0
    provider: str = "mock"
    tier: str | None = None
    alias: str | None = None


class BaseModelAdapter(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    def estimate_cost(self, total_tokens: int) -> float:
        return total_tokens / 1000.0 * self.config.cost_per_1k_tokens
