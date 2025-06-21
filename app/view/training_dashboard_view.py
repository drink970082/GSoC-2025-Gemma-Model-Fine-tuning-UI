import os
import time
from pathlib import Path

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
from backend.manager.global_manager import get_tensorboard_manager
from app.services.global_service import get_training_service
from config.app_config import get_config

config = get_config()


@st.fragment(run_every=1)
def update_tensorboard_event_loop():
    """Update TensorBoard data every second (separate from display fragments)."""
    get_tensorboard_manager().get_data()


@st.fragment(run_every=1)
def poll_training_status():
    """
    If training was active, this fragment checks if it has stopped.
    If so, it triggers a rerun to update the control panel to its terminal state.
    """
    training_service = get_training_service()
    if (
        not training_service.is_training_running()
        and st.session_state.session_started_by_app
    ):
        st.session_state.session_started_by_app = False
        st.rerun()


def show_training_dashboard_view():
    """Display the training interface."""
    st.title("LLM Fine-Tuning Dashboard")
    if "session_started_by_app" not in st.session_state:
        st.session_state.session_started_by_app = False
    poll_training_status()
    update_tensorboard_event_loop()
    # Dashboard Panels
    display_control_panel()
    st.divider()
    display_kpi_panel()
    st.divider()
    display_plots_panel()
    st.divider()
    display_system_usage_panel()
    st.divider()
    display_logs_panel()


if __name__ == "__main__":
    show_training_dashboard_view()
