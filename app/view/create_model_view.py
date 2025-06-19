import os
import time

import streamlit as st

from config.training_config import LOCK_FILE
from app.components.create_model.config_summary import (
    show_configuration_preview,
)
from app.components.create_model.data_source_selector import (
    show_data_source_section,
)
from app.components.create_model.tuning_method_selector import (
    show_fine_tuning_method_section,
)
from app.components.create_model.model_name_input import show_model_name_section
from app.components.create_model.model_selector import (
    show_model_selection_section,
)
from app.components.create_model.start_training_button import (
    show_start_training_section,
)
from backend.manager.global_manager import (
    get_process_manager,
    get_tensorboard_manager,
)


def show_create_model_view():
    """Display the create model interface."""
    # Initialize configuration dictionary
    config = {}

    # Get all configuration values
    st.subheader("1. Model Name")
    config["model_name"] = show_model_name_section()
    st.divider()
    st.subheader("2. Fine-tuning Method")
    config["method"], config["method_params"] = (
        show_fine_tuning_method_section()
    )
    st.divider()
    st.subheader("3. Data Source")
    config["data_source"], config["data_config"] = show_data_source_section()
    st.divider()
    st.subheader("4. Model Selection")
    config["model_config"] = show_model_selection_section()
    st.divider()
    st.subheader("5. Configuration Preview")
    show_configuration_preview(config)
    st.divider()
    st.subheader("6. Start Training")
    if show_start_training_section(config):
        # Start the training process
        process_manager = get_process_manager()
        tensorboard_manager = get_tensorboard_manager()
        tensorboard_manager.reset_training_time()
        process_manager.start_training(
            config["data_config"],
            config["model_config"],
        )

        # Set states to switch to the training dashboard
        st.session_state.session_started_by_app = True
        with st.spinner("Waiting for training process to initialize..."):
            start_time = time.time()
            while (
                not os.path.exists(LOCK_FILE) and time.time() - start_time < 10
            ):
                time.sleep(0.5)

        st.session_state.view = "training_dashboard"

        st.rerun()


if __name__ == "__main__":
    show_create_model_view()
