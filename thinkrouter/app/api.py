from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from thinkrouter.app.budgets import validate_budget
from thinkrouter.app.evaluators import get_evaluator
from thinkrouter.app.models import build_adapter, default_model_configs
from thinkrouter.app.router import JointPolicyEngine
from thinkrouter.app.schemas import ModelRequest, RunRequest, RunResponse, TraceRecord, model_to_dict
from thinkrouter.app.store import TraceStore

load_dotenv()

app = FastAPI(title="ThinkRouter", version="0.1.0")


def get_db_path() -> str:
    return os.getenv("THINKROUTER_DB_PATH", "results/traces/thinkrouter.sqlite")


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
        route = policy.route(request.query, request.task_type)
        model_id = route.model_id
        budget = route.budget
    validate_budget(budget)
    if model_id not in configs:
        configs[model_id] = configs[next(iter(configs))].__class__(model_id=model_id)
    config = configs[model_id]
    adapter = build_adapter(config)
    model_request = ModelRequest(
        query=request.query,
        task_type=request.task_type,
        model_id=model_id,
        budget=budget,
        metadata={"expected_answer": request.expected_answer},
    )
    model_response = adapter.generate(model_request)
    evaluation = get_evaluator(request.task_type).evaluate(model_response.output_text, request.expected_answer)
    trace = TraceRecord(
        query=request.query,
        task_type=request.task_type,
        selected_model=model_id,
        selected_budget=budget,
        output_text=model_response.output_text,
        score=evaluation.score,
        is_correct=evaluation.is_correct,
        expected_answer=evaluation.expected_answer,
        extracted_answer=evaluation.extracted_answer,
        prompt_tokens=model_response.prompt_tokens,
        completion_tokens=model_response.completion_tokens,
        total_tokens=model_response.total_tokens,
        cost_usd=adapter.estimate_cost(model_response.total_tokens),
        latency_s=model_response.latency_s,
        metadata={"route": model_to_dict(route) if route else None},
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
        "budgets": [0, 256, 1024, 4096],
    }

