import streamlit as st
from config.app_config import get_config
from config.dataclass import MethodConfig, LoraParams, DpoParams


config = get_config()


def show_fine_tuning_method_section() -> MethodConfig:
    """Display the fine-tuning method selection and parameters."""
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
    return MethodConfig(name=method, parameters=params)
