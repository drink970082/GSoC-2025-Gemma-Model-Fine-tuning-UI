import streamlit as st
import time
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
from app.components.create_model.tuning_method_selector import (
    show_fine_tuning_method_section,
)
from services.training_service import TrainingService
from config.dataclass import (
    TrainingConfig,
    MethodConfig,
    DataConfig,
    ModelConfig,
)


def _get_config(
    model_name: str,
    method_config: MethodConfig,
    data_config: DataConfig,
    model_config: ModelConfig,
) -> TrainingConfig:
    training_config = None
    if model_name and method_config and data_config and model_config:
        training_config = TrainingConfig(
            model_name=model_name,
            method_config=method_config,
            data_config=data_config,
            model_config=model_config,
        )
    return training_config


def show_create_model_view(training_service: TrainingService):
    """Display the create model interface."""
    config = {}
    st.subheader("1. Model Name")
    model_name = show_model_name_section()
    st.divider()
    st.subheader("2. Fine-tuning Method")
    method_config = show_fine_tuning_method_section()
    st.divider()
    st.subheader("3. Data Source")
    data_config = show_data_source_section()
    st.divider()
    st.subheader("4. Model Selection")
    model_config = show_model_selection_section()
    st.divider()
    training_config = _get_config(
        model_name, method_config, data_config, model_config
    )

    st.subheader("5. Configuration Preview")
    show_configuration_preview(training_config)
    st.divider()
    st.subheader("6. Start Training")
    if show_start_training_section(training_config):
        with st.spinner(
            "Waiting for training process to initialize...",
            show_time=True,
        ):
            training_service.start_training(training_config)
            st.session_state.session_started_by_app = True
            while not training_service.wait_for_lock_file():
                time.sleep(1)

        st.session_state.view = "training_dashboard"
        st.rerun()
