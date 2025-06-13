import json
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import treescope

from backend.data_pipeline import create_pipeline
from config.model_info import FINE_TUNING_METHODS, MODEL_INFO, TASK_TYPES


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
        list(FINE_TUNING_METHODS.keys()),
        help="Choose the fine-tuning approach based on your needs",
        horizontal=True,
    )

    # Show method description
    st.info(FINE_TUNING_METHODS[method]["description"])

    # Show advantages and disadvantages
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
    method_params = {}
    if method == "LoRA":
        st.markdown("#### LoRA Parameters")
        lora_params = FINE_TUNING_METHODS["LoRA"]["parameters"]
        method_params["lora_rank"] = st.number_input(
            "LoRA Rank",
            1,
            32,
            lora_params["lora_rank"]["default"],
            help=lora_params["lora_rank"]["description"],
        )
        method_params["lora_alpha"] = st.number_input(
            "LoRA Alpha",
            1,
            32,
            lora_params["lora_alpha"]["default"],
            help=lora_params["lora_alpha"]["description"],
        )
    elif method == "DPO":
        st.markdown("#### DPO Parameters")
        dpo_params = FINE_TUNING_METHODS["DPO"]["parameters"]
        method_params["dpo_beta"] = st.number_input(
            "DPO Beta",
            0.1,
            1.0,
            dpo_params["dpo_beta"]["default"],
            help=dpo_params["dpo_beta"]["description"],
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
        # Create pipeline and load data
        pipeline = create_pipeline(data_config)
        print(data_config)

        tab1, tab2 = st.tabs(["Raw Data Preview", "Tokenized Output Preview"])
        with tab1:
            st.subheader("Human-Readable Source Data")
            try:
                with st.spinner("Loading raw preview..."):
                    raw_examples = pipeline.get_raw_preview(num_records=5)
                    src_texts = [
                        text.decode("utf-8")
                        for text in raw_examples[
                            data_config["seq2seq_in_prompt"]
                        ]
                    ]
                    dst_texts = [
                        text.decode("utf-8")
                        for text in raw_examples[
                            data_config["seq2seq_in_response"]
                        ]
                    ]

                    df = pd.DataFrame(
                        {"Prompt": src_texts, "Response": dst_texts}
                    )
                    st.dataframe(
                        df,
                        use_container_width=True,
                        column_config={
                            "Prompt": st.column_config.TextColumn(
                                "Prompt", width="large"
                            ),
                            "Response": st.column_config.TextColumn(
                                "Response", width="large"
                            ),
                        },
                    )

            except Exception as e:
                st.error(e)

        with tab2:
            st.subheader("Model Input After Tokenization")
            try:
                with st.spinner("Loading tokenized preview..."):

                    tokenized_examples = pipeline.get_tokenized_preview(
                        num_records=5
                    )
                    # Show treescope visualization
                    with treescope.active_autovisualizer.set_scoped(
                        treescope.ArrayAutovisualizer()
                    ):
                        html = treescope.render_to_html(tokenized_examples)
                        components.html(html, height=250, scrolling=True)

                    # Show decoded text
                    with st.expander("Tokenizer Decoded Text", expanded=True):
                        if "input" in tokenized_examples:
                            # Create lists for the conversation
                            turns = []
                            for i in range(len(tokenized_examples["input"])):
                                # Decode the entire block
                                decoded_text = pipeline.tokenizer.decode(
                                    tokenized_examples["input"][i]
                                )

                                # Split into user and model parts while keeping the tokens
                                if (
                                    "<start_of_turn>user" in decoded_text
                                    and "<start_of_turn>model" in decoded_text
                                ):
                                    user_part = decoded_text.split(
                                        "<start_of_turn>model"
                                    )[0].strip()
                                    model_part = (
                                        "<start_of_turn>model"
                                        + decoded_text.split(
                                            "<start_of_turn>model"
                                        )[1].strip()
                                    )

                                    turns.append(
                                        {"user": user_part, "model": model_part}
                                    )

                            # Create DataFrame
                            df = pd.DataFrame(turns)

                            # Display in Streamlit
                            st.dataframe(
                                df,
                                use_container_width=True,
                                column_config={
                                    "user": st.column_config.TextColumn(
                                        "user", width="large"
                                    ),
                                    "model": st.column_config.TextColumn(
                                        "model", width="large"
                                    ),
                                },
                            )
            except Exception as e:
                st.error(e)

    return data_source, data_config


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

    # Show task type description
    # task = TASK_TYPES[task_type]
    # st.info(task["description"])
    # with st.expander("Task Use Cases"):
    #     for use_case in task["use_cases"]:
    #         st.markdown(f"- {use_case}")
    """Display the training parameters section."""
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


def show_configuration_preview(config):
    """Display the configuration preview section."""
    preview_config = {
        "Model Name": config["model_name"],
        "Fine-tuning Method": config["method"],
        "Method Parameters": config["method_params"],
        "Data Source": config["data_source"],
        "Data Configuration": config["data_config"],
        "Training Parameters": config["model_config"],
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
    config["model_config"] = show_model_selection_section()

    st.subheader("5. Configuration Preview")
    show_configuration_preview(config)

    st.subheader("6. Start Training")
    if show_start_training_section(config):
        # Store configuration in session state
        st.session_state.training_config = {
            "name": config["model_name"],
            "model": config["model_config"],
            "method": config["method"],
            "method_params": config["method_params"],
            "data_source": config["data_source"],
            "data_config": config["data_config"],
        }
        # Set page to training to trigger state change in Home.py
        st.session_state.training_active = True
        st.rerun()
