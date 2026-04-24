from __future__ import annotations

import pandas as pd
import streamlit as st

from thinkrouter.analytics import summarize_costs, summarize_latency


def render_dashboard(traces_df: pd.DataFrame) -> None:
    st.subheader("Dashboard")
    if traces_df.empty:
        st.info("No traces available.")
        return
    metrics = st.columns(4)
    metrics[0].metric("Traces", int(len(traces_df)))
    metrics[1].metric("Accuracy", f"{traces_df['is_correct'].astype(bool).mean():.3f}")
    metrics[2].metric("Avg Cost", f"${pd.to_numeric(traces_df['cost_usd'], errors='coerce').mean():.6f}")
    metrics[3].metric("P95 Latency", f"{pd.to_numeric(traces_df['latency_s'], errors='coerce').quantile(0.95):.2f}s")

    st.caption("Cost breakdown")
    st.dataframe(summarize_costs(traces_df), use_container_width=True)

    st.caption("Latency breakdown")
    st.dataframe(summarize_latency(traces_df), use_container_width=True)
