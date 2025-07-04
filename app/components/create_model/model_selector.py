import streamlit as st

from config.app_config import get_config
from config.dataclass import ModelConfig, LoraParams, DpoParams

config = get_config()


def show_model_selection_section() -> ModelConfig:
    """Display the model selection and task type section."""

    model_variant = st.selectbox(
        "Select Gemma Model",
        list(config.MODEL_INFO.keys()),
        index=0,
        help="Choose the model size based on your task and available resources",
    )

    # Model information
    model = config.MODEL_INFO[model_variant]
    st.info(
        f"""
        - Size: {model['size']}
        - {model['description']}
        """
    )

    # Show use cases and requirements
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

    method = st.radio(
        "Select Fine-tuning Method",
        list(config.FINE_TUNING_METHODS.keys()),
        help="Choose the fine-tuning approach based on your needs",
        horizontal=True,
    )

    # Show method description
    st.info(config.FINE_TUNING_METHODS[method]["description"])
    params: DpoParams | LoraParams | None = None
    # Show advantages and disadvantages
    with st.expander("Method Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Advantages:**")
            for adv in config.FINE_TUNING_METHODS[method]["advantages"]:
                st.markdown(f"- {adv}")
        with col2:
            st.markdown("**Disadvantages:**")
            for dis in config.FINE_TUNING_METHODS[method]["disadvantages"]:
                st.markdown(f"- {dis}")

    if method == "LoRA":
        st.markdown("#### LoRA Parameters")
        lora_params = config.FINE_TUNING_METHODS["LoRA"]["parameters"]
        lora_rank = st.number_input(
            "LoRA Rank",
            1,
            32,
            lora_params["lora_rank"]["default"],
            help=lora_params["lora_rank"]["description"],
        )
        params = LoraParams(lora_rank=lora_rank)
    elif method == "DPO":
        st.markdown("#### DPO Parameters")
        dpo_params = config.FINE_TUNING_METHODS["DPO"]["parameters"]
        dpo_beta = st.number_input(
            "DPO Beta",
            0.1,
            1.0,
            dpo_params["dpo_beta"]["default"],
            help=dpo_params["dpo_beta"]["description"],
        )
        params = DpoParams(dpo_beta=dpo_beta)

    st.subheader("Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input(
            "Number of Epochs",
            min_value=1,
            value=100,
            step=1,
            help="Enter the total number of training epochs.",
        )
    with col2:
        learning_rate = st.slider(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-3,
            value=1e-4,
            step=1e-6,
            format="%e",
            help="Set the learning rate. Small values like 1e-4 or 1e-5 are common.",
        )
    return ModelConfig(
        model_variant=model_variant,
        epochs=epochs,
        learning_rate=learning_rate,
        method=method,
        parameters=params,
    )
