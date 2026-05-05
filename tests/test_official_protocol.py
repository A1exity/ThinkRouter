from __future__ import annotations

from fastapi.testclient import TestClient

from thinkrouter.app.api import app
from thinkrouter.app.schemas import ModelResponse, RouteDecision
from thinkrouter.official_protocol import OFFICIAL_PROTOCOL


def test_official_protocol_is_frozen_to_expected_pool_and_benchmarks() -> None:
    assert OFFICIAL_PROTOCOL.model_pool == ("qwen-flash", "qwen-plus", "qwen-max")
    assert OFFICIAL_PROTOCOL.budgets == (0, 256, 1024)
    assert [item.benchmark for item in OFFICIAL_PROTOCOL.benchmarks] == ["gsm8k", "math500", "humaneval"]


def test_api_config_exposes_default_router_and_protocol_version() -> None:
    client = TestClient(app)
    response = client.get("/config")
    payload = response.json()

    assert response.status_code == 200
    assert payload["budgets"] == list(OFFICIAL_PROTOCOL.budgets)
    assert payload["default_router"] == OFFICIAL_PROTOCOL.default_router
    assert payload["official_protocol_version"] == OFFICIAL_PROTOCOL.version
    assert payload["model_pool"] == list(OFFICIAL_PROTOCOL.model_pool)
    assert set(payload["models"].keys()) == set(OFFICIAL_PROTOCOL.model_pool)
    for model_payload in payload["models"].values():
        assert "api_key" not in model_payload


def test_api_default_router_uses_phase2_stack(monkeypatch, tmp_path) -> None:
    import thinkrouter.app.api as api

    class FakeRouter:
        def route(self, query: str, task_type: str = "custom") -> RouteDecision:
            return RouteDecision(
                model_id="mock-cheap",
                budget=0,
                difficulty="easy",
                estimated_accuracy=0.9,
                estimated_cost=0.0,
                estimated_latency=0.0,
                explanation="phase2 default",
                router_name=OFFICIAL_PROTOCOL.default_router,
                route_confidence=0.8,
            )

    calls: list[str | None] = []

    def fake_build_runtime_router(model_configs, router_name: str | None = None):
        calls.append(router_name)
        return FakeRouter()

    def fail_legacy_policy(*args, **kwargs):
        raise AssertionError("legacy policy should not be constructed for default routed /run")

    def fake_generate_with_runtime(adapter, request):
        return ModelResponse(output_text="42", total_tokens=10), {"cache_hit": False}

    monkeypatch.setenv("THINKROUTER_DB_PATH", str(tmp_path / "api.sqlite"))
    monkeypatch.setenv("THINKROUTER_MODEL_POOL", "")
    monkeypatch.setenv("THINKROUTER_CHEAP_MODEL", "mock-cheap")
    monkeypatch.setenv("THINKROUTER_MID_MODEL", "")
    monkeypatch.setenv("THINKROUTER_STRONG_MODEL", "mock-strong")
    monkeypatch.delenv("THINKROUTER_OFFICIAL_RUNTIME_ROUTER", raising=False)
    monkeypatch.setattr(api, "build_runtime_router", fake_build_runtime_router)
    monkeypatch.setattr(api, "JointPolicyEngine", fail_legacy_policy)
    monkeypatch.setattr(api, "generate_with_runtime", fake_generate_with_runtime)

    client = TestClient(app)
    response = client.post(
        "/run",
        json={
            "query": "What is 40 + 2?",
            "task_type": "gsm8k",
            "expected_answer": "42",
            "use_router": True,
        },
    )
    payload = response.json()

    assert response.status_code == 200
    assert calls == [OFFICIAL_PROTOCOL.default_router]
    assert payload["route"]["router_name"] == OFFICIAL_PROTOCOL.default_router
    assert payload["trace"]["metadata"]["router_name"] == OFFICIAL_PROTOCOL.default_router
