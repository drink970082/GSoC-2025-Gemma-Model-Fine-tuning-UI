import streamlit as st

from config.model_info import MODEL_INFO, TASK_TYPES


def show_model_selection_section():
    """Display the model selection and task type section."""
    model_config = {}
    model_config["model_variant"] = st.selectbox(
        "Select Gemma Model",
        list(MODEL_INFO.keys()),
        index=0,
        help="Choose the model size based on your task and available resources",
    )

    # Model information
    model = MODEL_INFO[model_config["model_variant"]]
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

    task_type = st.selectbox(
        "Select Task Type",
        list(TASK_TYPES.keys()),
    )

    # Training parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        model_config["epochs"] = st.slider("Number of Epochs", 1, 10, 5)
    with col2:
        model_config["batch_size"] = st.slider("Batch Size", 1, 32, 8)
    with col3:
        model_config["learning_rate"] = st.slider(
            "Learning Rate", 1e-6, 1e-3, 1e-4
        )
    return model_config
