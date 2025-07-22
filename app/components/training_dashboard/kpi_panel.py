import streamlit as st

from services.training_service import TrainingService


@st.fragment(run_every=1)
def display_kpi_panel(training_service: TrainingService) -> None:
    """Display the key performance indicators panel."""
    if st.session_state.abort_training:
        kpi_data = st.session_state.frozen_kpi_data
    else:
        kpi_data = training_service.get_kpi_data()
        st.session_state.frozen_kpi_data = kpi_data

    has_metadata = kpi_data.get("total_params")
    is_training = kpi_data.get("current_step", 0) > 0

    if not has_metadata and not is_training:
        st.info("Waiting for training data...")
        return

    if has_metadata:
        _create_metadata_panel(kpi_data)
    if is_training:
        _create_training_progress_panel(kpi_data)
        _create_performance_metrics_panel(kpi_data)


def _format_time(seconds: int) -> str:
    """Format seconds into HH:MM:SS format."""
    if seconds <= 0:
        return "00:00:00"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _create_metadata_panel(kpi_data: dict) -> None:
    """Create the metadata panel."""
    st.markdown("#### Model Information")
    metadata_cols = st.columns(3)
    metadata_cols[0].metric(
        "Total Parameters", f"{kpi_data.get('total_params', 0):,d}"
    )
    metadata_cols[1].metric(
        "Total Memory (GB)",
        round(kpi_data.get("total_bytes", 0) / 1024**3, 4),
    )
    metadata_cols[2].metric(
        "Transformer Layers", kpi_data.get("total_layers", 0)
    )


def _create_training_progress_panel(kpi_data: dict) -> None:
    """Create the training progress panel."""
    st.markdown("#### Training Progress")
    kpi_cols = st.columns(4)
    total_steps = int(kpi_data.get("total_steps", 0))
    current_step = int(kpi_data.get("current_step", 0))
    step_str = (
        f"{current_step}/{total_steps}"
        if total_steps > 0
        else str(int(current_step))
    )
    training_time_seconds = int(kpi_data.get("training_time", 0) * 3600)
    training_time_str = _format_time(training_time_seconds)
    kpi_cols[0].metric("Global Step", step_str)
    kpi_cols[1].metric(
        "Current Loss", f"{kpi_data.get('current_loss', 0.0):.4f}"
    )
    kpi_cols[2].metric(
        "Training Speed",
        f"{kpi_data.get('training_speed', 0.0):.2f} steps/sec",
    )
    kpi_cols[3].metric("Training Time", training_time_str)


def _create_performance_metrics_panel(kpi_data: dict) -> None:
    """Create the performance metrics panel."""
    st.markdown("#### Performance Metrics")
    perf_cols = st.columns(4)
    perf_cols[0].metric(
        "Data Throughput",
        f"{kpi_data.get('data_throughput', 0.0):.0f} tokens/sec",
    )
    perf_cols[1].metric("ETA", kpi_data.get("eta_str", "N/A"))
    perf_cols[2].metric(
        "Avg Step Time", f"{kpi_data.get('avg_step_time', 0.0):.3f}s"
    )
    perf_cols[3].metric(
        "Avg Eval Time", f"{kpi_data.get('avg_eval_time', 0.0):.3f}s"
    )


