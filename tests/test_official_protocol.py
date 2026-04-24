from __future__ import annotations

from fastapi.testclient import TestClient

from thinkrouter.app.api import app
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
    assert payload["default_router"] == OFFICIAL_PROTOCOL.default_router
    assert payload["official_protocol_version"] == OFFICIAL_PROTOCOL.version
