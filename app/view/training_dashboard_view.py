import streamlit as st
from app.components.training_dashboard.control_panel import (
    display_control_panel,
)
from app.components.training_dashboard.kpi_panel import display_kpi_panel
from app.components.training_dashboard.logs_panel import display_logs_panel
from app.components.training_dashboard.plots_panel import display_plots_panel
from app.components.training_dashboard.system_usage_panel import (
    display_system_usage_panel,
)
from services.training_service import TrainingService


@st.fragment(run_every=1)
def poll_training_status(training_service: TrainingService) -> None:
    """
    If training was active, this fragment checks if it has stopped.
    If so, it triggers a rerun to update the control panel to its terminal state.
    """
    status = training_service.is_training_running()

    if st.session_state.abort_training:
        return

    if (
        status in ["FINISHED", "FAILED"]
    ) and st.session_state.session_started_by_app:
        st.session_state.session_started_by_app = False
        st.rerun()

    training_service.poll_system_usage()
    training_service.get_tensorboard_data()

    if status == "RUNNING":
        st.info(f"Training in progress.")


def show_training_dashboard_view(training_service: TrainingService) -> None:
    """Display the training interface."""
    st.title("LLM Fine-Tuning Dashboard")
    poll_training_status(training_service)

    # Dashboard Panels
    display_control_panel(training_service)
    st.divider()
    st.subheader("Key Performance Indicators")
    display_kpi_panel(training_service)
    st.divider()
    st.subheader("Core Performance Plots")
    display_plots_panel(training_service)
    st.divider()
    st.subheader("System Resource Usage")
    display_system_usage_panel(training_service)
    st.divider()
    display_logs_panel(training_service)
