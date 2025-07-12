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
    cpu_history = system_history.get("CPU Utilization (%)", pd.DataFrame())
    if len(cpu_history) < MIN_DATA_POINTS:
        st.info("Collecting system usage data...")
        return

    # Filter charts based on available data and GPU status
    available_charts = {
        label: data
        for label, data in system_history.items()
        if not data.empty and ("GPU" not in label or training_service.has_gpu())
    }
    
    if not available_charts:
        st.info("No system data available to display.")
        return

    # Display charts
    chart_columns = st.columns(len(available_charts))
    for i, (label, data) in enumerate(available_charts.items()):
        with chart_columns[i]:
            st.markdown(f"**{label}**")
            st.line_chart(data, use_container_width=True)
