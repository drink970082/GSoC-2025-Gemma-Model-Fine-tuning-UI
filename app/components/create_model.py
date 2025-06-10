import json
import tempfile

import streamlit as st
import streamlit.components.v1 as components
import treescope

from backend.data_pipeline import create_pipeline


def show_model_name_section():
    """Display the model name input section."""
    return st.text_input(
        "Enter a name for your model",
        placeholder="e.g., gemma-2b-chat-finetuned",
        value="gemma-2b-chat-finetuned",
    )


def show_fine_tuning_method_section():
    """Display the fine-tuning method selection and parameters."""
    method = st.radio(
        "Select Fine-tuning Method",
        ["Standard", "LoRA", "DPO"],
        help="Choose the fine-tuning approach based on your needs",
        horizontal=True,
    )

    method_params = {}
    if method == "LoRA":
        st.markdown("#### LoRA Parameters")
        method_params["lora_rank"] = st.number_input(
            "LoRA Rank",
            1,
            32,
            8,
            help="Rank of the LoRA update matrices (higher = more capacity)",
        )
        method_params["lora_alpha"] = st.number_input(
            "LoRA Alpha", 1, 32, 16, help="Alpha parameter for LoRA scaling"
        )
    elif method == "DPO":
        st.markdown("#### DPO Parameters")
        method_params["dpo_beta"] = st.number_input(
            "DPO Beta",
            0.1,
            1.0,
            0.1,
            help="Temperature parameter for DPO (higher = more exploration)",
        )

    return method, method_params


def show_data_source_section():
    """Display the data source selection and configuration."""
    data_source = st.radio(
        "Select Data Source",
        ["HuggingFace Dataset", "TensorFlow Dataset", "Custom JSON Upload"],
    )

    # Common data options
    data_config = {}

    if data_source == "HuggingFace Dataset":
        data_config["source"] = "huggingface"
        data_config["dataset_name"] = st.text_input(
            "Dataset Name",
            placeholder="e.g., google/fleurs, open-r1/Mixture-of-Thoughts",
            value="fka/awesome-chatgpt-prompts",
        )
        data_config["dataset_config"] = st.text_input(
            "Dataset Config",
            help="Optional: Specify dataset-specific config (like language/domain). Leave empty for default 'main' config.",
            placeholder="e.g., hi_in, code",
        )
        data_config["split"] = st.text_input(
            "Split (e.g., 'train', 'train[:80%]', 'train[80%:]')",
            help="Optional: Specify dataset split. Leave empty for default 'train' split.",
            placeholder="e.g., train, train[:80%], train[80%:]",
        )

    elif data_source == "TensorFlow Dataset":
        data_config["source"] = "tensorflow"
        data_config["dataset_name"] = st.text_input(
            "TensorFlow Dataset Name",
            placeholder="e.g., mtnt, mtnt/en-fr",
            value="mtnt",
        )
        data_config["split"] = st.text_input(
            "Split (e.g., 'train', 'train[:80%]', 'train[80%:]')",
            value="train",
            help="Optional: Specify dataset split. Leave empty for default 'train' split.",
            placeholder="e.g., train, train[:80%], train[80%:]",
        )

    else:  # Custom JSON Upload
        data_config["source"] = "json"
        uploaded_file = st.file_uploader(
            "Upload JSON file",
            type=["json"],
            help="Upload a JSON file containing your training data",
        )
        if uploaded_file:
            temp_file_path = None
            try:
                # Read the uploaded file line by line, decoding each line
                # The file is opened in text mode by Streamlit, so we can iterate over lines directly.
                # We need to ensure the file content is properly decoded.
                stringio = uploaded_file.getvalue().decode("utf-8")
                lines = stringio.splitlines()
                data = []
                for line in lines:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
                # Save the list of JSON objects as a proper JSON array in a temporary file
                with tempfile.NamedTemporaryFile(
                    mode="w+",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as temp_file:
                    json.dump(data, temp_file)
                    data_config["data_path"] = temp_file.name

            except json.JSONDecodeError as e:
                st.error(
                    f"Error parsing JSONL file: Invalid JSON on one of the lines. Details: {e}"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")

    data_config["shuffle"] = st.checkbox(
        "Shuffle Dataset",
        value=True,
        help="Whether to shuffle the dataset before training",
    )

    # Universal Sequence-to-Sequence Parameters
    with st.expander(
        "Configure Sequence-to-Sequence Parameters", expanded=True
    ):
        col1, col2 = st.columns(2)
        with col1:
            data_config["seq2seq_in_prompt"] = st.text_input(
                "Source Field Name",
                value="src",
                help="Field name for source text (prompt) in the dataset",
            )
            data_config["seq2seq_max_length"] = st.number_input(
                "Maximum Sequence Length",
                min_value=1,
                max_value=512,
                value=200,
                help="Maximum length of input sequences",
            )
        with col2:
            data_config["seq2seq_in_response"] = st.text_input(
                "Target Field Name",
                value="dst",
                help="Field name for target text (response) in the dataset",
            )
            data_config["seq2seq_truncate"] = st.checkbox(
                "Truncate Long Sequences",
                value=True,
                help="Whether to truncate sequences longer than max_length",
            )
    # Dataset Preview Section
    if st.button("Preview Dataset", type="secondary"):
        try:
            # Create pipeline and load data
            with st.spinner("Loading dataset...", show_time=True):
                # Create pipeline and load data
                print(data_config)
                pipeline = create_pipeline(data_config)
                pipeline.load_data()

            sample = pipeline.data[0]

            if sample:

                # Show treescope visualization
                with st.expander("Data Structure Visualization", expanded=True):
                    with treescope.active_autovisualizer.set_scoped(
                        treescope.ArrayAutovisualizer()
                    ):
                        html = treescope.render_to_html(sample)
                        components.html(html, height=250, scrolling=True)

                # Show decoded text
                with st.expander("Tokenizer Decoded Text", expanded=True):
                    if "input" in sample:
                        input_text = pipeline.tokenizer.decode(
                            sample["input"][0]
                        )
                        st.text_area("Input Text", input_text, height=100)
            else:
                st.warning("Dataset is empty or could not be loaded")

        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
            st.error("Please check your configuration and try again")

    return data_source, data_config


def show_model_selection_section():
    """Display the model selection and task type section."""
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

    task_type = st.selectbox(
        "Select Task Type",
        ["Text Generation", "Question Answering", "Chat", "Custom"],
    )

    return model_variant, task_type


def show_training_parameters_section():
    """Display the training parameters section."""
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Number of Epochs", 1, 10, 5)
    with col2:
        batch_size = st.slider("Batch Size", 1, 32, 8)
    return epochs, batch_size


def show_configuration_preview(config):
    """Display the configuration preview section."""
    preview_config = {
        "Model Name": config["model_name"],
        "Fine-tuning Method": config["method"],
        "Method Parameters": config["method_params"],
        "Data Source": config["data_source"],
        "Data Configuration": config["data_config"],
        "Model Variant": config["model_variant"],
        "Task Type": config["task_type"],
        "Training Parameters": {
            "Epochs": config["epochs"],
            "Batch Size": config["batch_size"],
        },
    }

    with st.expander("Review Your Configuration", expanded=True):
        st.json(preview_config)


def show_start_training_section(config):
    """Display the start training section and handle validation."""
    if st.button("Start Fine-tuning", type="primary"):
        if not config["model_name"]:
            st.error("Please enter a model name")
        elif config["data_source"] == "HuggingFace Dataset" and (
            not config["data_config"].get("org")
            or not config["data_config"].get("dataset_name")
        ):
            st.error("Please provide both organization and dataset name")
        elif config["data_source"] == "TensorFlow Dataset" and not config[
            "data_config"
        ].get("dataset_name"):
            st.error("Please provide dataset name")
        elif config["data_source"] == "Custom JSON Upload" and not config[
            "data_config"
        ].get("file"):
            st.error("Please upload a JSON file")
        else:
            return True
    return False


def show_create_model():
    """Display the create model interface."""
    # Initialize configuration dictionary
    config = {}

    # Get all configuration values
    st.subheader("1. Model Name")
    config["model_name"] = show_model_name_section()

    st.subheader("2. Fine-tuning Method")
    config["method"], config["method_params"] = (
        show_fine_tuning_method_section()
    )

    st.subheader("3. Data Source")
    config["data_source"], config["data_config"] = show_data_source_section()

    st.subheader("4. Model Selection")
    config["model_variant"], config["task_type"] = (
        show_model_selection_section()
    )

    st.subheader("5. Training Parameters")
    config["epochs"], config["batch_size"] = show_training_parameters_section()

    st.subheader("6. Configuration Preview")
    show_configuration_preview(config)

    st.subheader("7. Start Training")
    if show_start_training_section(config):
        # Store configuration in session state
        st.session_state.training_config = {
            "name": config["model_name"],
            "model": config["model_variant"],
            "task": config["task_type"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "method": config["method"],
            "method_params": config["method_params"],
            "data_source": config["data_source"],
            "data_config": config["data_config"],
        }
        # Set page to training to trigger state change in Home.py
        st.session_state.training_active = True
        st.rerun()
