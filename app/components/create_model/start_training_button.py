import streamlit as st

from backend.data_pipeline import create_pipeline
from config.dataclass import TrainingConfig


def show_start_training_section(config: TrainingConfig) -> bool:
    """Display the start training section and handle validation."""
    if st.button(
        "Start Fine-tuning", type="primary", key="start_training_button"
    ):
        if not config:
            st.error("Please enter all the fields.")
            return False

        # Model name validation
        if not config.model_name or not config.model_name.strip():
            st.error("Please enter a model name")
            return False

        # Data config validation
        if (
            not config.data_config.dataset_name
            or not config.data_config.dataset_name.strip()
        ):
            if config.data_config.source == "json":
                st.error("Please upload a JSON file")
            else:
                st.error("Please provide dataset name")
            return False

        # Additional validations
        if config.data_config.batch_size <= 0:
            st.error("Batch size must be greater than 0")
            return False

        if config.data_config.seq2seq_max_length <= 0:
            st.error("Maximum sequence length must be greater than 0")
            return False

        if (
            not config.data_config.seq2seq_in_prompt
            or not config.data_config.seq2seq_in_prompt.strip()
        ):
            st.error("Please provide a prompt field name")
            return False

        if (
            not config.data_config.seq2seq_in_response
            or not config.data_config.seq2seq_in_response.strip()
        ):
            st.error("Please provide a response field name")
            return False

        # Model config validation
        if config.model_config.epochs <= 0:
            st.error("Number of epochs must be greater than 0")
            return False

        if config.model_config.learning_rate <= 0:
            st.error("Learning rate must be greater than 0")
            return False

        # Method-specific validation
        if (
            config.model_config.method == "LoRA"
            and config.model_config.parameters
        ):
            if config.model_config.parameters.lora_rank <= 0:
                st.error("LoRA rank must be greater than 0")
                return False

        try:
            pipeline = create_pipeline(config.data_config)
            pipeline.get_train_dataset()
        except Exception as e:
            st.error(f"Error creating pipeline: {e}")
            return False
        return True
