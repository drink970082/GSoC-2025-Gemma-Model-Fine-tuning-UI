import streamlit as st
from typing import Union, Tuple
from config.model_info import MODEL_INFO
from config.fine_tuning_info import FINE_TUNING_METHODS
from config.dataclass import ModelConfig, LoraParams, DpoParams

# Constants
DEFAULT_MODEL_INDEX = 3
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-2
MIN_LORA_RANK = 1
MAX_LORA_RANK = 32
MIN_DPO_BETA = 0.1
MAX_DPO_BETA = 1.0


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


def _create_dpo_parameters() -> DpoParams:
    """Create DPO parameters from user input."""
    st.markdown("#### DPO Parameters")
    dpo_config = FINE_TUNING_METHODS["DPO"]["parameters"]
    dpo_beta = st.number_input(
        "DPO Beta",
        MIN_DPO_BETA,
        MAX_DPO_BETA,
        dpo_config["dpo_beta"]["default"],
        help=dpo_config["dpo_beta"]["description"],
    )
    return DpoParams(dpo_beta=dpo_beta)


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
    st.info(FINE_TUNING_METHODS[method]["description"])

    with st.expander("Method Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Advantages:**")
            for adv in FINE_TUNING_METHODS[method]["advantages"]:
                st.markdown(f"- {adv}")
        with col2:
            st.markdown("**Disadvantages:**")
            for dis in FINE_TUNING_METHODS[method]["disadvantages"]:
                st.markdown(f"- {dis}")


def _create_method_parameters(
    method: str,
) -> Union[LoraParams, DpoParams, None]:
    """Create parameters based on the selected method."""
    if method == "LoRA":
        return _create_lora_parameters()
    elif method == "DPO":
        return _create_dpo_parameters()
    return None


def _get_training_parameters() -> Tuple[int, float]:
    """Get training parameters from user input."""
    st.markdown("#### Training Parameters")
    col1, col2 = st.columns(2)

    with col1:
        epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            value=DEFAULT_EPOCHS,
            step=1,
            help="Enter the total number of training epochs",
        )

    with col2:
        learning_rate = st.slider(
            "Learning Rate",
            min_value=MIN_LEARNING_RATE,
            max_value=MAX_LEARNING_RATE,
            value=DEFAULT_LEARNING_RATE,
            step=1e-6,
            format="%e",
            help="Set the learning rate. Small values like 1e-4 or 1e-5 are common",
        )

    return epochs, learning_rate


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
    epochs, learning_rate = _get_training_parameters()
    
    return ModelConfig(
        model_variant=model_variant,
        epochs=epochs,
        learning_rate=learning_rate,
        method=method,
        parameters=params,
    )
