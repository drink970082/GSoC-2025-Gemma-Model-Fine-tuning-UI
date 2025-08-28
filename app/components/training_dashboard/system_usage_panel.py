import pandas as pd
import streamlit as st
from services.training_service import TrainingService

# Constants
MIN_DATA_POINTS = 2

@st.fragment(run_every=1)
def display_system_usage_panel(training_service: TrainingService) -> None:
    """Display the system resource usage panel."""
    if not training_service.has_gpu():
        st.warning("NVIDIA GPU not detected. GPU monitoring is disabled.")
    system_history = training_service.get_system_usage_history()
    # Check if we have enough data
    available_charts = {
        label: data
        for label, data in system_history.items()
        if not data.empty
        and len(data) >= MIN_DATA_POINTS
        and ("GPU" not in label or training_service.has_gpu())
    }
    if not available_charts:
        st.info("Collecting system usage data...")
        return

    # Display charts
    chart_columns = st.columns(len(available_charts))
    for i, (label, data) in enumerate(available_charts.items()):
        with chart_columns[i]:
            st.markdown(f"**{label}**")
            st.line_chart(data, use_container_width=True)
