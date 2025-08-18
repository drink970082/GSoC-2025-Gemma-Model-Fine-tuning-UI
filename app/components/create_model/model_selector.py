from typing import Tuple, Union

import streamlit as st

from config.dataclass import DpoParams, LoraParams, ModelConfig
from config.fine_tuning_info import FINE_TUNING_METHODS
from config.model_info import MODEL_INFO

# Constants
DEFAULT_MODEL_INDEX = 3
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1e-2
MIN_LORA_RANK = 1
MAX_LORA_RANK = 32


def show_model_selection_section() -> ModelConfig:
    """Display the model selection and task type section."""

    # Model selection
    model_variant = st.selectbox(
        "Select Gemma Model",
        list(MODEL_INFO.keys()),
        index=DEFAULT_MODEL_INDEX,
        help="Choose the model size based on your task and available resources",
    )
    _display_model_info(model_variant)

    # Fine-tuning method selection
    method = st.radio(
        "Select Fine-tuning Method",
        list(FINE_TUNING_METHODS.keys()),
        help="Choose the fine-tuning approach based on your needs",
        horizontal=True,
    )
    _display_method_info(method)

    # Create method parameters
    params = _create_method_parameters(method)

    # Get training parameters
    epochs, learning_rate = _get_training_parameters(method)

    return ModelConfig(
        model_variant=model_variant,
        epochs=epochs,
        learning_rate=learning_rate,
        method=method,
        parameters=params,
    )


def _create_lora_parameters() -> LoraParams:
    """Create LoRA parameters from user input."""
    st.markdown("#### LoRA Parameters")
    lora_config = FINE_TUNING_METHODS["LoRA"]["parameters"]
    lora_rank = st.number_input(
        "LoRA Rank",
        MIN_LORA_RANK,
        MAX_LORA_RANK,
        lora_config["lora_rank"]["default"],
        help=lora_config["lora_rank"]["description"],
    )
    return LoraParams(lora_rank=lora_rank)


def _display_model_info(model_variant: str) -> None:
    """Display model information."""
    model = MODEL_INFO[model_variant]
    st.info(f"- Size: {model['size']}\n- {model['description']}")

    with st.expander("Model Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Use Cases:**")
            for use_case in model["use_cases"]:
                st.markdown(f"- {use_case}")
        with col2:
            st.markdown("**Requirements:**")
            for req, value in model["requirements"].items():
                st.markdown(f"- {req}: {value}")


def _display_method_info(method: str) -> None:
    """Display method information."""
    info = FINE_TUNING_METHODS[method]
    st.info(info["description"])

    with st.expander("Method Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Advantages:**")
            for adv in info["advantages"]:
                st.markdown(f"- {adv}")
            st.markdown("**Best For:**")
            for use in info.get("best_for", []):
                st.markdown(f"- {use}")
        with col2:
            st.markdown("**Disadvantages:**")
            for dis in info["disadvantages"]:
                st.markdown(f"- {dis}")
            st.markdown("**Summary:**")
            st.markdown(f"- Memory Usage: {info.get('memory_usage', 'N/A')}")
            st.markdown(
                f"- Training Speed: {info.get('training_speed', 'N/A')}"
            )
            st.markdown(f"- Use Case: {info.get('use_case', 'N/A')}")


def _create_method_parameters(
    method: str,
) -> Union[LoraParams, DpoParams, None]:
    """Create parameters based on the selected method."""
    if method == "LoRA":
        return _create_lora_parameters()
    return None


def _get_training_parameters(method: str) -> Tuple[int, float]:
    """Get training parameters from user input."""
    st.markdown("#### Training Parameters")
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            value=FINE_TUNING_METHODS[method]["default_parameters"]["epochs"],
            step=1,
            help="Enter the total number of training epochs",
        )

    with col2:
        learning_rate = st.number_input(
            "Learning Rate",
            min_value=MIN_LEARNING_RATE,
            max_value=MAX_LEARNING_RATE,
            value=FINE_TUNING_METHODS[method]["default_parameters"][
                "learning_rate"
            ],
            step=MIN_LEARNING_RATE,
            format="%.3e",
            help="Set the learning rate. Small values like 1e-4 or 1e-5 are common",
        )

    return epochs, round(learning_rate, 6)
