import streamlit as st

from config.dataclass import TrainingConfig


def show_start_training_section(config: TrainingConfig):
    """Display the start training section and handle validation."""
    if st.button("Start Fine-tuning", type="primary"):
        if not config.model_name:
            st.error("Please enter a model name")
        elif not config.data_config.dataset_name:
            if (
                config.data_config.source == "HuggingFace Dataset"
                or config.data_config.source == "TensorFlow Dataset"
            ):
                st.error("Please provide dataset name")
            elif config.data_config.source == "Custom JSON Upload":
                st.error("Please upload a JSON file")
            else:
                st.error("Please provide dataset name")
        else:
            return True
    return False
