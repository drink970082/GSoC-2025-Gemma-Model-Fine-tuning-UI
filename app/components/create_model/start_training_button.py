import streamlit as st

from backend.data_pipeline import create_pipeline
from config.dataclass import TrainingConfig


def _disable_button(config: TrainingConfig) -> None:
    validation_passed = _validate_config(config)
    st.session_state["validation_passed"] = validation_passed
    return validation_passed


def _validate_config(config: TrainingConfig) -> bool:
    if not config:
        st.session_state["validation_error"] = "Please enter all the fields."
        return False

    # Model name validation
    if not config.model_name or not config.model_name.strip():
        st.session_state["validation_error"] = "Please enter a model name"
        return False

    # Data config validation
    if (
        not config.data_config.dataset_name
        or not config.data_config.dataset_name.strip()
    ):
        if config.data_config.source == "json":
            st.session_state["validation_error"] = "Please upload a JSON file"
        else:
            st.session_state["validation_error"] = "Please provide dataset name"
        return False

    # Additional validations
    if config.data_config.batch_size <= 0:
        st.session_state["validation_error"] = "Batch size must be greater than 0"
        return False

    if config.data_config.seq2seq_max_length <= 0:
        st.session_state["validation_error"] = "Maximum sequence length must be greater than 0"
        return False

    if (
        not config.data_config.seq2seq_in_prompt
        or not config.data_config.seq2seq_in_prompt.strip()
    ):
        st.session_state["validation_error"] = "Please provide a prompt field name"
        return False

    if (
        not config.data_config.seq2seq_in_response
        or not config.data_config.seq2seq_in_response.strip()
    ):
        st.session_state["validation_error"] = "Please provide a response field name"
        return False

    # Model config validation
    if config.model_config.epochs <= 0:
        st.session_state["validation_error"] = "Number of epochs must be greater than 0"
        return False

    if config.model_config.learning_rate <= 0:
        st.session_state["validation_error"] = "Learning rate must be greater than 0"
        return False

    # Method-specific validation
    if config.model_config.method == "LoRA" and config.model_config.parameters:
        if config.model_config.parameters.lora_rank <= 0:
            st.session_state["validation_error"] = "LoRA rank must be greater than 0"
            return False

    try:
        pipeline = create_pipeline(config.data_config)
        pipeline.get_train_dataset()
    except Exception as e:
        st.session_state["validation_error"] = f"Error creating dataset pipeline: {e}"
        return False

    return True


def show_start_training_section(config: TrainingConfig) -> bool:
    """Display the start training section and handle validation."""
    if st.button(
        "Start Fine-tuning",
        type="primary",
        key="start_training_button",
        on_click=_disable_button,
        args=(config,),
        disabled=st.session_state.get("validation_passed", False),
    ):
        if not st.session_state.get("validation_passed", False):
            st.error(st.session_state.get("validation_error", ""))
            return False
        else:
            return True
