import streamlit as st

from config.app_config import get_config

config = get_config()


def show_model_selection_section():
    """Display the model selection and task type section."""
    model_config = {}
    model_config["model_variant"] = st.selectbox(
        "Select Gemma Model",
        list(config.MODEL_INFO.keys()),
        index=0,
        help="Choose the model size based on your task and available resources",
    )

    # Model information
    model = config.MODEL_INFO[model_config["model_variant"]]
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

    # task_type = st.selectbox(
    #     "Select Task Type",
    #     list(TASK_TYPES.keys()),
    # )

    # Training parameters
    st.subheader("Training Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        model_config["epochs"] = st.number_input(
            "Number of Epochs",
            min_value=1,
            value=100,
            step=1,
            help="Enter the total number of training epochs.",
        )
    with col2:
        model_config["batch_size"] = st.slider(
            "Batch Size",
            min_value=1,
            max_value=32,
            value=4,
            help="Select the number of samples to process in each batch.",
        )
    with col3:
        model_config["learning_rate"] = st.slider(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-3,
            value=1e-4,
            step=1e-6,
            format="%e",
            help="Set the learning rate. Small values like 1e-4 or 1e-5 are common.",
        )
    return model_config
