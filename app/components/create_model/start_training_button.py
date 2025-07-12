import streamlit as st
from config.dataclass import TrainingConfig


def show_start_training_section(config: TrainingConfig) -> bool:
    """Display the start training section and handle validation."""
    if st.button(
        "Start Fine-tuning", type="primary", key="start_training_button"
    ):
        if not config.model_name or not config.model_name.strip():
            st.error("Please enter a model name")
            return False

        if (
            not config.data_config.dataset_name
            or not config.data_config.dataset_name.strip()
        ):
            if config.data_config.source == "json":
                st.error("Please upload a JSON file")
            else:
                st.error("Please provide dataset name")
            return False
        return True
