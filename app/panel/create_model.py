import os
import time

import streamlit as st

from config.training_config import (
    LOCK_FILE,
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
)
from app.components.create_model.config_preview import (
    show_configuration_preview,
)
from app.components.create_model.data_source import show_data_source_section
from app.components.create_model.fine_tuning import (
    show_fine_tuning_method_section,
)
from app.components.create_model.model_name import show_model_name_section
from app.components.create_model.model_selection import (
    show_model_selection_section,
)
from app.components.create_model.start_training import (
    show_start_training_section,
)
from app.utils.tensorboard import reset_tensorboard_training_time
from backend.utils.manager.GlobalManager import get_process_manager


def show_create_model_panel():
    """Display the create model interface."""
    # Initialize configuration dictionary
    config = {}

    # Get all configuration values
    st.subheader("1. Model Name")
    config["model_name"] = show_model_name_section()

    st.subheader("2. Fine-tuning Method")
    config["method"], config["method_params"] = (
        show_fine_tuning_method_section()
    )

    st.subheader("3. Data Source")
    config["data_source"], config["data_config"] = show_data_source_section()

    st.subheader("4. Model Selection")
    config["model_config"] = show_model_selection_section()

    st.subheader("5. Configuration Preview")
    show_configuration_preview(config)

    st.subheader("6. Start Training")
    if show_start_training_section(config):
        # Store configuration in session state
        st.session_state.training_config = {
            "name": config["model_name"],
            "model": config["model_config"],
            "method": config["method"],
            "method_params": config["method_params"],
            "data_source": config["data_source"],
            "data_config": config["data_config"],
        }

        # Start the training process
        process_manager = get_process_manager()
        reset_tensorboard_training_time()
        process_manager.start_training(
            config["data_config"],
            config["model_config"],
        )
        print("--------------------------------")
        print("config")
        print(config["data_config"])
        print(config["model_config"])
        print("--------------------------------")
        print("DEFAULT_DATA_CONFIG")
        print(DEFAULT_DATA_CONFIG)
        print(DEFAULT_MODEL_CONFIG)
        print("--------------------------------")

        # Set states to switch to the training dashboard
        st.session_state.session_started_by_app = True
        with st.spinner("Waiting for training process to initialize..."):
            start_time = time.time()
            while (
                not os.path.exists(LOCK_FILE) and time.time() - start_time < 10
            ):
                time.sleep(0.5)

        st.session_state.view = "training"

        st.rerun()


if __name__ == "__main__":
    show_create_model_panel()
