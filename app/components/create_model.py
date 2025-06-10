import json

import streamlit as st
from components.training import show_training


def show_create_model():
    """Display the create model interface."""

    # Model name
    st.subheader("1. Model Name")
    model_name = st.text_input("Enter a name for your model")

    # Fine-tuning method selection
    st.subheader("2. Fine-tuning Method")
    method = st.radio(
        "Select Fine-tuning Method",
        ["Standard", "LoRA", "DPO"],
        help="Choose the fine-tuning approach based on your needs",
        horizontal=True,
    )

    # Show method-specific parameters
    method_params = {}
    if method == "LoRA":
        st.markdown("#### LoRA Parameters")
        method_params["lora_rank"] = st.number_input("LoRA Rank", 1, 32, 8)
        method_params["lora_alpha"] = st.number_input("LoRA Alpha", 1, 32, 16)
    elif method == "DPO":
        st.markdown("#### DPO Parameters")
        method_params["dpo_beta"] = st.number_input("DPO Beta", 0.1, 1.0, 0.1)

    # Data source
    st.subheader("3. Data Source")
    data_source = st.radio(
        "Select Data Source",
        ["HuggingFace Dataset", "TensorFlow Dataset", "Custom JSON Upload"],
    )

    # Common data options
    data_config = {
        "shuffle": st.checkbox(
            "Shuffle Dataset",
            value=True,
            help="Whether to shuffle the dataset before training"
        )
    }

    if data_source == "HuggingFace Dataset":
        data_config["org"] = st.text_input("Organization/User Name")
        data_config["dataset_name"] = st.text_input("Dataset Name")
        data_config["split"] = st.text_input(
            "Split (e.g., 'train', 'train[:80%]', 'train[80%:]')",
            help="Optional: Specify dataset split. Leave empty for default 'train' split."
        )
        if data_config["org"] and data_config["dataset_name"]:
            st.info(
                f"Will load dataset from: {data_config['org']}/{data_config['dataset_name']}"
            )

    elif data_source == "TensorFlow Dataset":
        data_config["dataset_name"] = st.text_input("TensorFlow Dataset Name")
        data_config["split"] = st.text_input(
            "Split (e.g., 'train', 'train[:80%]', 'train[80%:]')",
            help="Optional: Specify dataset split. Leave empty for default 'train' split."
        )
        if data_config["dataset_name"]:
            st.info(f"Will load dataset: {data_config['dataset_name']}")

    else:  # Custom JSON Upload
        uploaded_file = st.file_uploader("Upload JSON file", type=["json"])
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                st.success("File uploaded successfully!")
                data_config["file"] = uploaded_file.name

                # Show data preview
                with st.expander("Data Preview", expanded=True):
                    if isinstance(data, list):
                        st.write(f"Number of examples: {len(data)}")
                        if len(data) > 0:
                            st.json(data[0])
                    else:
                        st.json(data)
            except json.JSONDecodeError:
                st.error("Invalid JSON file")

    # Model selection
    st.subheader("4. Model Selection")
    model_variant = st.selectbox(
        "Select Gemma Model",
        ["Gemma 2B", "Gemma 7B"],
        help="Choose the model size based on your task and available resources",
    )

    # Model information
    if model_variant == "Gemma 2B":
        st.info(
            """
        - Size: 2 billion parameters
        - Best for: Smaller tasks, limited GPU memory
        - Training time: ~2-4 hours on single GPU
        """
        )
    else:
        st.info(
            """
        - Size: 7 billion parameters
        - Best for: Complex tasks, better performance
        - Training time: ~6-8 hours on single GPU
        """
        )

    # Task selection
    task_type = st.selectbox(
        "Select Task Type",
        ["Text Generation", "Question Answering", "Chat", "Custom"],
    )

    # Training parameters
    st.subheader("5. Training Parameters")
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Number of Epochs", 1, 10, 5)
    with col2:
        batch_size = st.slider("Batch Size", 1, 32, 8)

    # Preview Section
    st.subheader("6. Configuration Preview")
    preview_config = {
        "Model Name": model_name,
        "Fine-tuning Method": method,
        "Method Parameters": method_params,
        "Data Source": data_source,
        "Data Configuration": data_config,
        "Model Variant": model_variant,
        "Task Type": task_type,
        "Training Parameters": {"Epochs": epochs, "Batch Size": batch_size},
    }

    # Show preview in expandable section
    with st.expander("Review Your Configuration", expanded=True):
        st.json(preview_config)

    # Start Training Button
    st.subheader("7. Start Training")
    if st.button("Start Fine-tuning", type="primary"):
        if not model_name:
            st.error("Please enter a model name")
        elif data_source == "HuggingFace Dataset" and (
            not data_config.get("org") or not data_config.get("dataset_name")
        ):
            st.error("Please provide both organization and dataset name")
        elif data_source == "TensorFlow Dataset" and not data_config.get(
            "dataset_name"
        ):
            st.error("Please provide dataset name")
        elif data_source == "Custom JSON Upload" and not data_config.get(
            "file"
        ):
            st.error("Please upload a JSON file")
        else:
            # Store configuration in session state
            st.session_state.training_config = {
                "name": model_name,
                "model": model_variant,
                "task": task_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "method": method,
                "method_params": method_params,
                "data_source": data_source,
                "data_config": data_config,
            }
            # Set page to training to trigger state change in Home.py
            st.session_state.training_active = True
            st.rerun()
