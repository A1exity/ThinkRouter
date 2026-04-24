from __future__ import annotations

import pandas as pd
import streamlit as st


def render_route_inspector(traces_df: pd.DataFrame) -> None:
    st.subheader("Route Inspector")
    if traces_df.empty:
        st.info("No traces available.")
        return
    trace_ids = traces_df["id"].astype(str).tolist() if "id" in traces_df.columns else [str(index) for index in traces_df.index]
    selected_trace_id = st.selectbox("Trace id", trace_ids)
    if "id" in traces_df.columns:
        trace = traces_df[traces_df["id"].astype(str) == selected_trace_id].iloc[0]
    else:
        trace = traces_df.iloc[int(selected_trace_id)]
    left, right = st.columns(2)
    left.json(
        {
            "task_type": trace.get("task_type"),
            "selected_model": trace.get("selected_model"),
            "selected_budget": trace.get("selected_budget"),
            "route_confidence": trace.get("route_confidence"),
            "fallback_triggered": trace.get("fallback_triggered"),
            "fallback_reason": trace.get("fallback_reason"),
            "is_correct": trace.get("is_correct"),
        }
    )
    right.json(
        {
            "selected_model_provider": trace.get("selected_model_provider"),
            "selected_model_tier": trace.get("selected_model_tier"),
            "cost_usd": trace.get("cost_usd"),
            "latency_s": trace.get("latency_s"),
            "error_type": trace.get("error_type"),
        }
    )
    st.text_area("Query", str(trace.get("query", "")), height=120)
    st.text_area("Output", str(trace.get("output_text", "")), height=220)
