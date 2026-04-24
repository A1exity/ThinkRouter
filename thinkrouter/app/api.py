from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from thinkrouter.app.budgets import budget_to_dict, compile_budget_config, validate_budget
from thinkrouter.app.evaluators import get_evaluator
from thinkrouter.app.models import build_adapter, default_model_configs
from thinkrouter.app.router import JointPolicyEngine, available_router_names, build_runtime_router
from thinkrouter.app.schemas import EvalResult, ModelRequest, ModelResponse, RunRequest, RunResponse, TraceRecord, model_to_dict
from thinkrouter.official_protocol import OFFICIAL_PROTOCOL
from thinkrouter.app.store import TraceStore
from thinkrouter.runtime import generate_with_runtime

load_dotenv()

app = FastAPI(title="ThinkRouter", version="0.1.0")


def get_db_path() -> str:
    return os.getenv("THINKROUTER_DB_PATH", "results/traces/thinkrouter.sqlite")


def get_default_router_name() -> str:
    return os.getenv("THINKROUTER_OFFICIAL_RUNTIME_ROUTER", OFFICIAL_PROTOCOL.default_router)


def get_runtime() -> tuple[dict, TraceStore, JointPolicyEngine]:
    configs = default_model_configs()
    store = TraceStore(get_db_path())
    policy = JointPolicyEngine(list(configs.keys()))
    return configs, store, policy


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run_query(request: RunRequest) -> RunResponse:
    configs, store, policy = get_runtime()
    route = None
    model_id = request.model_id
    budget = request.budget
    if request.use_router:
        router_name = request.router_name or get_default_router_name()
        if router_name == "legacy_joint_policy":
            route = policy.route(request.query, request.task_type)
        else:
            router = build_runtime_router(list(configs.values()), router_name)
            route = router.route(request.query, request.task_type)
        model_id = route.model_id
        budget = route.budget
    validate_budget(budget)
    budget_config = compile_budget_config(budget)
    if model_id not in configs:
        configs[model_id] = configs[next(iter(configs))].__class__(model_id=model_id)
    config = configs[model_id]
    adapter = build_adapter(config)
    candidate_models = request.candidate_models or list(configs.keys())
    model_request = ModelRequest(
        query=request.query,
        task_type=request.task_type,
        model_id=model_id,
        budget=budget,
        budget_config=budget_config,
        candidate_set_signature="|".join(sorted(candidate_models)),
        metadata={"expected_answer": request.expected_answer},
    )
    model_response, runtime_meta = generate_with_runtime(adapter, model_request)
    if model_response is None:
        model_response = ModelResponse(
            output_text="",
            parsed_output=None,
            finish_reason="error",
            error_type=str(runtime_meta["error_type"]),
            provider_meta={
                "provider": config.provider,
                "tier": config.tier,
                "budget_id": budget_config.budget_id,
                "cache_hit": runtime_meta["cache_hit"],
                "error_message": runtime_meta["error_message"],
            },
            raw_response={"error_message": runtime_meta["error_message"], "budget": budget_to_dict(budget_config)},
        )
        evaluation = EvalResult(
            score=0.0,
            is_correct=False,
            parsed_output=None,
            extracted_answer=None,
            expected_answer=request.expected_answer,
            error_type=str(runtime_meta["error_type"]),
            judge_metadata={"runtime_error": runtime_meta["error_message"]},
        )
    else:
        evaluation = get_evaluator(request.task_type).evaluate(model_response.output_text, request.expected_answer, model_request.metadata)
    trace = TraceRecord(
        query_id=None,
        benchmark=request.task_type,
        query=request.query,
        query_text=request.query,
        task_type=request.task_type,
        selected_model=model_id,
        selected_model_provider=config.provider,
        selected_model_tier=config.tier,
        selected_model_alias=config.alias,
        selected_budget=budget,
        selected_budget_id=budget_config.budget_id,
        budget_config=budget_to_dict(budget_config),
        output_text=model_response.output_text,
        parsed_output=evaluation.parsed_output or model_response.parsed_output,
        score=evaluation.score,
        is_correct=evaluation.is_correct,
        expected_answer=evaluation.expected_answer,
        extracted_answer=evaluation.extracted_answer,
        judge_metadata=evaluation.judge_metadata,
        prompt_tokens=model_response.prompt_tokens,
        completion_tokens=model_response.completion_tokens,
        total_tokens=model_response.total_tokens,
        cost_usd=adapter.estimate_cost(model_response.total_tokens),
        latency_s=model_response.latency_s,
        error_type=evaluation.error_type or model_response.error_type,
        route_reason=route.explanation if route else None,
        candidate_set_signature=model_request.candidate_set_signature,
        prompt_template_version=budget_config.prompt_template_version,
        provider_response_meta=model_response.provider_meta,
        route_confidence=route.route_confidence if route else None,
        fallback_triggered=route.fallback_triggered if route else False,
        fallback_reason=route.fallback_reason if route else None,
        metadata={
            "route": model_to_dict(route) if route else None,
            "router_name": request.router_name or (route.router_name if route else None),
            "cache_hit": runtime_meta["cache_hit"],
            "selected_model_provider": config.provider,
            "selected_model_tier": config.tier,
            "selected_model_alias": config.alias,
        },
    )
    trace = store.insert_trace(trace)
    return RunResponse(route=route, model_response=model_response, evaluation=evaluation, trace=trace)


@app.get("/traces", response_model=list[TraceRecord])
def list_traces(limit: int = 100) -> list[TraceRecord]:
    return TraceStore(get_db_path()).list_traces(limit=limit)


@app.get("/config")
def config() -> dict[str, object]:
    configs = default_model_configs()
    return {
        "db_path": str(Path(get_db_path())),
        "models": {key: value.__dict__ for key, value in configs.items()},
        "model_pool": list(configs.keys()),
        "budgets": [0, 256, 1024, 4096],
        "routers": available_router_names(),
        "default_router": get_default_router_name(),
        "official_protocol_version": OFFICIAL_PROTOCOL.version,
    }

