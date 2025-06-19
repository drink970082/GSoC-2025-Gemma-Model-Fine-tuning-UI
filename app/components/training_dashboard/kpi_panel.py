import time

import streamlit as st

from backend.manager.global_manager import get_tensorboard_manager
from config.training_config import DEFAULT_MODEL_CONFIG


@st.fragment(run_every=1)
def display_kpis():
    """Display the key performance indicators panel."""
    st.subheader("Key Performance Indicators")

    # Get TensorBoard manager
    manager = get_tensorboard_manager()

    # Get all data from manager
    metadata = manager.get_metadata()
    latest_values = manager.get_latest_values()

    if not metadata and not latest_values:
        st.info("Waiting for training data...")
        return

    # Display metadata in first row
    if metadata:
        st.markdown("#### Model Information")
        meta_cols = st.columns(3)

        # Column 1: Model Size
        if "num_params" in metadata and metadata["num_params"]:
            meta_cols[0].metric(
                "Model Parameters", f"{metadata['num_params']:,}"
            )

        # Column 2: Memory Usage
        if "parameters" in metadata and metadata["parameters"]:
            param_summary = manager.parse_parameter_summary(
                str(metadata["parameters"])
            )
            if param_summary.get("total_bytes"):
                memory_mb = param_summary["total_bytes"] / 1024 / 1024
                meta_cols[1].metric("Model Memory", f"{memory_mb:.1f} MB")

        # Column 3: Model Architecture
        if "parameters" in metadata and metadata["parameters"]:
            param_summary = manager.parse_parameter_summary(
                str(metadata["parameters"])
            )
            if param_summary.get("layers"):
                meta_cols[2].metric(
                    "Transformer Layers", f"{len(param_summary['layers'])}"
                )

    # Display training KPIs
    st.markdown("#### Training Progress")
    kpi_cols = st.columns(4)

    # Get model config for total steps
    model_config = DEFAULT_MODEL_CONFIG
    total_steps = int(model_config.get("epochs", 0))

    # Get all values from manager
    latest_step = manager.get_current_step()
    current_loss = manager.get_current_loss()
    training_speed = manager.get_training_speed()
    training_time = manager.get_training_time()
    data_throughput = manager.get_data_throughput()
    avg_step_time = manager.get_avg_step_time()
    eval_time = manager.get_avg_eval_time()

    # Calculate ETA
    eta_str = "N/A"
    if training_speed > 0 and total_steps > 0:
        remaining_steps = total_steps - latest_step
        if remaining_steps > 0:
            eta_seconds = remaining_steps / training_speed
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

    # Display KPIs
    kpi_cols[0].metric("Global Step", f"{int(latest_step)}/{total_steps}")
    kpi_cols[1].metric("Current Loss", f"{current_loss:.4f}")
    kpi_cols[2].metric("Training Speed", f"{training_speed:.2f} steps/sec")
    kpi_cols[3].metric("Training Time", f"{training_time:.2f} hours")

    # Second row of KPIs
    st.markdown("#### Performance Metrics")
    perf_cols = st.columns(4)

    perf_cols[0].metric("Data Throughput", f"{data_throughput:.0f} pts/sec")
    perf_cols[1].metric("ETA", eta_str)
    perf_cols[2].metric("Avg Step Time", f"{avg_step_time:.3f}s")
    perf_cols[3].metric("Avg Eval Time", f"{eval_time:.3f}s")
