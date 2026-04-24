from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from thinkrouter.app.api import config as api_config
from thinkrouter.app.api import run_query
from thinkrouter.app.budgets import BUDGET_LEVELS
from thinkrouter.app.models import default_model_configs
from thinkrouter.app.schemas import RunRequest, model_to_dict
from thinkrouter.app.store import TraceStore
from thinkrouter.experiments.sample_data import load_day1_samples

load_dotenv()

st.set_page_config(page_title="ThinkRouter", layout="wide")
st.title("ThinkRouter")
st.caption("Adaptive thinking-budget routing for verifiable reasoning tasks")

configs = default_model_configs()
model_ids = list(configs.keys())
router_names = [None] + list(api_config()["routers"])
samples = load_day1_samples()
config_rows = pd.DataFrame(
    [
        {
            "model_id": config.model_id,
            "provider": config.provider,
            "tier": config.tier,
            "alias": config.alias,
            "backend": config.backend,
            "cost_per_1k_tokens": config.cost_per_1k_tokens,
        }
        for config in configs.values()
    ]
)

with st.sidebar:
    st.header("Run")
    use_router = st.checkbox("Use router", value=False)
    selected_router = st.selectbox("Router", router_names, index=0, disabled=not use_router, format_func=lambda value: "legacy_joint_policy" if value is None else str(value))
    selected_model = st.selectbox("Model", model_ids, disabled=use_router)
    selected_budget = st.selectbox("Budget", list(BUDGET_LEVELS), index=0, disabled=use_router)
    db_path = st.text_input("SQLite DB", os.getenv("THINKROUTER_DB_PATH", "results/traces/thinkrouter.sqlite"))

sample_labels = [f"{sample.sample_id}: {sample.query[:60]}" for sample in samples]
selected_label = st.selectbox("Sample", sample_labels)
selected_sample = samples[sample_labels.index(selected_label)]
query = st.text_area("Query", selected_sample.query, height=120)
expected_answer = st.text_input("Expected answer", selected_sample.expected_answer)

if st.button("Run query", type="primary"):
    os.environ["THINKROUTER_DB_PATH"] = db_path
    response = run_query(
        RunRequest(
            query=query,
            task_type="gsm8k",
            expected_answer=expected_answer,
            model_id=selected_model,
            budget=int(selected_budget),
            use_router=use_router,
            router_name=selected_router,
        )
    )
    if response.route:
        st.subheader("Route")
        st.json(model_to_dict(response.route))
        route_metrics = st.columns(4)
        route_metrics[0].metric("Router", response.route.router_name or "legacy_joint_policy")
        route_metrics[1].metric("Confidence", "-" if response.route.route_confidence is None else f"{response.route.route_confidence:.3f}")
        route_metrics[2].metric("Fallback", "yes" if response.route.fallback_triggered else "no")
        route_metrics[3].metric("Fallback reason", response.route.fallback_reason or "-")
    st.subheader("Model output")
    st.write(response.model_response.output_text)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Correct", "yes" if response.evaluation.is_correct else "no")
    metric_cols[1].metric("Score", f"{response.evaluation.score:.2f}")
    metric_cols[2].metric("Cost", f"${response.trace.cost_usd:.6f}")
    metric_cols[3].metric("Latency", f"{response.trace.latency_s:.2f}s")

st.divider()
st.subheader("Model Pool")
st.dataframe(config_rows, use_container_width=True)

st.divider()
st.subheader("Recent traces")
try:
    traces = TraceStore(db_path).list_traces(limit=100)
    if traces:
        df = pd.DataFrame([model_to_dict(trace) for trace in traces])
        st.dataframe(
            df[
                [
                    "id",
                    "task_type",
                    "selected_model",
                    "selected_model_provider",
                    "selected_model_tier",
                    "selected_budget",
                    "route_confidence",
                    "fallback_triggered",
                    "fallback_reason",
                    "is_correct",
                    "cost_usd",
                    "latency_s",
                    "query",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No traces yet. Run a query or execute the Day-1 grid script.")
except Exception as exc:
    st.warning(f"Could not load traces: {exc}")

