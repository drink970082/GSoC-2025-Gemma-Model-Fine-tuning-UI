from typing import Optional

import streamlit as st

from app.components.create_model.config_summary import (
    show_configuration_preview,
)
from app.components.create_model.data_source_selector import (
    show_data_source_section,
)
from app.components.create_model.model_name_input import show_model_name_section
from app.components.create_model.model_selector import (
    show_model_selection_section,
)
from app.components.create_model.start_training_button import (
    show_start_training_section,
)
from config.dataclass import DataConfig, ModelConfig, TrainingConfig
from services.training_service import TrainingService


def _get_config(
    model_name: str,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> Optional[TrainingConfig]:
    """Create a training configuration from the model name, data config, and model config."""
    if model_name and data_config and model_config:
        return TrainingConfig(
            model_name=model_name,
            data_config=data_config,
            model_config=model_config,
        )
    return None


def _reset_session_state() -> None:
    """Reset the session state to its initial values."""
    st.session_state["abort_training"] = False
    st.session_state["session_started_by_app"] = True
    st.session_state["frozen_kpi_data"] = {}
    st.session_state["frozen_log"] = "No logs available."
    st.session_state["frozen_loss_metrics"] = {}
    st.session_state["frozen_perf_metrics"] = {}


def _handle_training_start(
    training_service: TrainingService, training_config: TrainingConfig
) -> None:
    """Handle training start process."""
    with st.spinner(
        "Waiting for training process to initialize...", show_time=True
    ):
        _reset_session_state()
        if training_service.start_training(training_config):
            st.session_state["view"] = "training_dashboard"
            st.rerun()
        else:
            st.error("Training failed to start. Please try again.")


def show_create_model_view(training_service: TrainingService) -> None:
    """Display the create model interface."""
    st.title("Create Model")
    st.subheader("1. Model Name")
    model_name = show_model_name_section()
    st.divider()
    st.subheader("2. Data Source")
    data_config = show_data_source_section()
    st.divider()
    st.subheader("3. Model Selection")
    model_config = show_model_selection_section()
    st.divider()
    training_config = _get_config(model_name, data_config, model_config)
    st.subheader("4. Configuration Preview")
    show_configuration_preview(training_config)
    st.divider()
    st.subheader("5. Start Training")
    if show_start_training_section(training_config):
        _handle_training_start(training_service, training_config)
