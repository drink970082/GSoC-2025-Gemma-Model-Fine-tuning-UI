import pandas as pd
import psutil
import pynvml
import streamlit as st
from services.training_service import TrainingService


@st.fragment(run_every=1)
def display_system_usage_panel(training_service: TrainingService):
    """Display the system resource usage panel."""
    st.subheader("System Resource Usage")
    if not training_service.has_gpu():
        st.warning("NVIDIA GPU not detected. GPU monitoring is disabled.")

    history_dfs = training_service.get_system_usage_history()

    cpu_history = history_dfs.get("CPU Utilization (%)", pd.DataFrame())
    if len(cpu_history) < 2:
        st.info("Collecting system usage data...")
        return

    # Dynamically create columns for available charts
    charts_to_show = {
        label: df
        for label, df in history_dfs.items()
        if not df.empty and ("GPU" not in label or training_service.has_gpu())
    }

    chart_cols = st.columns(len(charts_to_show))

    for i, (label, df) in enumerate(charts_to_show.items()):
        with chart_cols[i]:
            st.markdown(f"**{label}**")
            st.line_chart(df, use_container_width=True)
