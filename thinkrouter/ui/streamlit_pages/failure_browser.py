from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from thinkrouter.analytics import build_failure_browser_frame


def render_failure_browser(default_csv: str) -> None:
    st.subheader("Failure Browser")
    csv_path = st.text_input("Failure source CSV", default_csv, key="failure_source_csv")
    if not csv_path or not Path(csv_path).exists():
        st.info("Choose a completed grid CSV to browse failures.")
        return
    failures = build_failure_browser_frame(csv_path)
    if failures.empty:
        st.success("No failures found in the selected CSV.")
        return

    selected_error = st.selectbox("Error type", ["all"] + sorted(failures["error_type"].astype(str).unique().tolist()))
    selected_model = st.selectbox("Model", ["all"] + sorted(failures["selected_model"].astype(str).unique().tolist()))

    filtered = failures.copy()
    if selected_error != "all":
        filtered = filtered[filtered["error_type"].astype(str) == selected_error]
    if selected_model != "all":
        filtered = filtered[filtered["selected_model"].astype(str) == selected_model]

    st.dataframe(
        filtered[
            [
                "sample_id",
                "task_type",
                "selected_model",
                "selected_budget",
                "error_type",
                "cost_usd",
                "latency_s",
                "output_preview",
            ]
        ],
        use_container_width=True,
    )

    selected_sample = st.selectbox("Failure sample", filtered["sample_id"].astype(str).tolist(), key="failure_sample")
    detail = filtered[filtered["sample_id"].astype(str) == selected_sample].iloc[0]
    st.code(str(detail["query"]))
    st.text_area("Full output", str(detail["output_text"]), height=220)
