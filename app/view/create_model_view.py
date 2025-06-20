import os
import time

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
from app.components.create_model.tuning_method_selector import (
    show_fine_tuning_method_section,
)
from backend.manager.global_manager import (
    get_process_manager,
    get_tensorboard_manager,
)
from config.app_config import get_config

config = get_config()


def show_create_model_view():
    """Display the create model interface."""
    # Initialize configuration dictionary
    create_model_config = {}

    # Get all configuration values
    st.subheader("1. Model Name")
    create_model_config["model_name"] = show_model_name_section()
    st.divider()
    st.subheader("2. Fine-tuning Method")
    create_model_config["method"], create_model_config["method_params"] = (
        show_fine_tuning_method_section()
    )
    st.divider()
    st.subheader("3. Data Source")
    create_model_config["data_source"], create_model_config["data_config"] = (
        show_data_source_section()
    )
    st.divider()
    st.subheader("4. Model Selection")
    create_model_config["model_config"] = show_model_selection_section()
    st.divider()
    st.subheader("5. Configuration Preview")
    show_configuration_preview(create_model_config)
    st.divider()
    st.subheader("6. Start Training")
    if show_start_training_section(create_model_config):
        # Start the training process
        process_manager = get_process_manager()
        tensorboard_manager = get_tensorboard_manager()
        tensorboard_manager.reset_training_time()
        process_manager.update_config(
            config["data_config"], config["model_config"]
        )
        process_manager.start_training()

        # Set states to switch to the training dashboard
        st.session_state.session_started_by_app = True
        st.session_state.model_config = create_model_config["model_config"]
        with st.spinner("Waiting for training process to initialize..."):
            start_time = time.time()
            while (
                not os.path.exists(config.LOCK_FILE)
                and time.time() - start_time < 10
            ):
                time.sleep(0.5)

        st.session_state.view = "training_dashboard"

        st.rerun()


if __name__ == "__main__":
    show_create_model_view()
