import streamlit as st


def show_import_model():
    """Display the import model interface."""
    st.title("Import Existing Model")

    # Model upload
    st.header("1. Upload Model Checkpoint")
    uploaded_file = st.file_uploader(
        "Upload model checkpoint", type=["pt", "pth", "bin"]
    )

    if uploaded_file:
        st.success("Model checkpoint uploaded successfully!")

    # Model configuration
    st.header("2. Model Configuration")
    model_name = st.text_input(
        "Model Name", placeholder="Enter a name for your model"
    )
    model_variant = st.selectbox(
        "Base Model",
        ["Gemma 2B", "Gemma 7B"],
        help="Select the base model variant",
    )

    # Model information
    st.header("3. Model Information")
    task_type = st.selectbox(
        "Task Type",
        ["Text Generation", "Question Answering", "Chat", "Custom"],
    )

    # Import button
    if st.button("Import Model", type="primary"):
        if not model_name:
            st.error("Please enter a model name")
        elif not uploaded_file:
            st.error("Please upload a model checkpoint")
        else:
            # Store model info
            st.session_state.trained_models.append(
                {
                    "name": model_name,
                    "status": "success",
                    "date": "Imported",
                    "base_model": model_variant,
                    "task": task_type,
                }
            )

            st.success("Model imported successfully!")
            st.session_state.training_active = True
            st.rerun()
