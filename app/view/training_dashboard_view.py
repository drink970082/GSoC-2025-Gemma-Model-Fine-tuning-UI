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
from config.app_config import get_config, TrainingStatus

config = get_config()


@st.fragment(run_every=1)
def poll_training_status(training_service: TrainingService):
    """
    If training was active, this fragment checks if it has stopped.
    If so, it triggers a rerun to update the control panel to its terminal state.
    """
    if st.session_state.abort_training:
        return
    if (
        training_service.is_training_running() == TrainingStatus.FINISHED
        or training_service.is_training_running() == TrainingStatus.FAILED
    ) and st.session_state.session_started_by_app:
        st.session_state.session_started_by_app = False
        st.rerun()

    status = training_service.get_training_status()
    training_service.poll_system_usage()
    training_service.get_tensorboard_data()
    st.info(f"Training in progress... Status: {status}")


def show_training_dashboard_view(training_service: TrainingService):
    """Display the training interface."""
    st.title("LLM Fine-Tuning Dashboard")
    if "session_started_by_app" not in st.session_state:
        st.session_state.session_started_by_app = False
    if "abort_training" not in st.session_state:
        st.session_state.abort_training = False
    poll_training_status(training_service)
    # Dashboard Panels
    display_control_panel(training_service)
    st.divider()
    display_kpi_panel(training_service)
    st.divider()
    display_plots_panel(training_service)
    st.divider()
    display_system_usage_panel(training_service)
    st.divider()
    display_logs_panel(training_service)
