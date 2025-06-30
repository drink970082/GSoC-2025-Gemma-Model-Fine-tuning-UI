import time
import streamlit as st
from services.training_service import TrainingService


@st.fragment(run_every=1)
def display_kpi_panel(training_service: TrainingService):
    """Display the key performance indicators panel."""
    st.subheader("Key Performance Indicators")
    kpi_data = training_service.get_kpi_data()
    has_metadata = kpi_data.get("total_params") or kpi_data.get(
        "total_memory_mb"
    )
    is_training = kpi_data.get("current_step", 0) > 0

    if not has_metadata or not is_training:
        st.info("Waiting for training data...")
        return

    if is_training:
        st.markdown("#### Training Progress")
        kpi_cols = st.columns(4)
        total_steps = kpi_data.get("total_steps", 0)
        current_step = kpi_data.get("current_step", 0)
        step_str = (
            f"{current_step}/{total_steps}"
            if total_steps > 0
            else str(current_step)
        )

        kpi_cols[0].metric("Global Step", step_str)
        kpi_cols[1].metric(
            "Current Loss", f"{kpi_data.get('current_loss', 0.0):.4f}"
        )
        kpi_cols[2].metric(
            "Training Speed",
            f"{kpi_data.get('training_speed', 0.0):.2f} steps/sec",
        )
        kpi_cols[3].metric(
            "Training Time", kpi_data.get("training_time", "00:00:00")
        )

        st.markdown("#### Performance Metrics")
        perf_cols = st.columns(4)
        perf_cols[0].metric(
            "Data Throughput",
            f"{kpi_data.get('data_throughput', 0.0):.0f} samples/sec",
        )
        perf_cols[1].metric("ETA", kpi_data.get("eta_str", "N/A"))
        perf_cols[2].metric(
            "Avg Step Time", f"{kpi_data.get('avg_step_time', 0.0):.3f}s"
        )
        perf_cols[3].metric(
            "Avg Eval Time", f"{kpi_data.get('avg_eval_time', 0.0):.3f}s"
        )

    elif has_metadata:
        # If we have metadata but training hasn't started, show a clear message.
        st.info("Waiting for training to start to show progress KPIs...")
