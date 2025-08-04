from __future__ import annotations
import json
import tempfile
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import treescope
from streamlit.runtime.uploaded_file_manager import UploadedFile
from config.dataclass import DataConfig

# Constants
DEFAULT_BATCH_SIZE = 4
MAX_BATCH_SIZE = 32
DEFAULT_SEQ_LENGTH = 200
MAX_SEQ_LENGTH = 512
PREVIEW_RECORDS = 5


def show_data_source_section() -> DataConfig:
    """Display the data source selection and configuration."""
    # Data source selection
    data_source = st.radio(
        "Select Data Source",
        ["HuggingFace Dataset", "TensorFlow Dataset", "Custom JSON Upload"],
        key="data_source_selector",
    )

    # Get source-specific configuration
    source, dataset_name, dataset_config, split = _get_source_config(
        data_source
    )

    # Common configuration
    shuffle, batch_size = _get_common_config()

    # Sequence-to-sequence configuration
    seq2seq_config = _get_seq2seq_config()

    # Create data config
    data_config = DataConfig(
        source=source,
        dataset_name=dataset_name,
        config=dataset_config,
        split=split.strip() if split else "train",
        shuffle=shuffle,
        batch_size=batch_size,
        seq2seq_in_prompt=seq2seq_config["prompt_field"],
        seq2seq_in_response=seq2seq_config["response_field"],
        seq2seq_max_length=seq2seq_config["max_length"],
        seq2seq_truncate=seq2seq_config["truncate"],
    )

    # Dataset preview
    _show_dataset_preview(data_config)

    return data_config


def _get_source_config(data_source: str) -> Tuple[str, str, Optional[str], str]:
    """Get configuration for the selected data source."""
    dataset_name = ""
    dataset_config = None
    split = "train"
    print(data_source)

    if data_source == "HuggingFace Dataset":
        source = "huggingface"
        dataset_name = st.text_input(
            "Dataset Name",
            placeholder="e.g., google/fleurs, open-r1/Mixture-of-Thoughts",
            key="dataset_name_input",
        )
        dataset_config = st.text_input(
            "Dataset Config",
            help="Optional: Specify dataset-specific config (like language/domain). Leave empty for default 'main' config.",
            placeholder="e.g., hi_in, code",
            key="dataset_config_input",
        )
        split = st.text_input(
            "Split",
            help="Optional: Specify dataset split. Leave empty for default 'train' split.",
            placeholder="e.g., train, train[:80%], train[80%:]",
            key="split_input",
        )

    elif data_source == "TensorFlow Dataset":
        source = "tensorflow"
        dataset_name = st.text_input(
            "TensorFlow Dataset Name",
            placeholder="e.g., mtnt, mtnt/en-fr",
            key="dataset_name_input",
        )
        split = st.text_input(
            "Split",
            help="Optional: Specify dataset split. Leave empty for default 'train' split.",
            placeholder="e.g., train, train[:80%], train[80%:]",
            key="split_input",
        )

    else:  # Custom JSON Upload
        source = "json"
        uploaded_file = st.file_uploader(
            "Upload JSON file",
            type=["json"],
            help="Upload a JSON file containing your training data",
            key="uploaded_file_input",
        )
        if uploaded_file:
            dataset_name = _process_uploaded_json(uploaded_file)

    return source, dataset_name, dataset_config, split


def _process_uploaded_json(uploaded_file: UploadedFile) -> str:
    """Process uploaded JSON file and return temporary file path."""
    try:
        content = uploaded_file.getvalue().decode("utf-8")
        lines = content.splitlines()
        data = [json.loads(line) for line in lines if line.strip()]

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False, encoding="utf-8"
        ) as temp_file:
            json.dump(data, temp_file)
            return temp_file.name

    except json.JSONDecodeError as e:
        st.error(
            f"Error parsing JSON file: Invalid JSON on one of the lines. Details: {e}"
        )
        return ""
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        return ""


def _get_common_config() -> Tuple[bool, int]:
    """Get common configuration options."""
    col1, col2 = st.columns(2)

    with col1:
        shuffle = st.checkbox(
            "Shuffle Dataset",
            value=True,
            help="Whether to shuffle the dataset before training",
            key="shuffle_checkbox",
        )

    with col2:
        batch_size = st.slider(
            "Batch Size",
            min_value=1,
            max_value=MAX_BATCH_SIZE,
            value=DEFAULT_BATCH_SIZE,
            help="Select the number of samples to process in each batch",
            key="batch_size_slider",
        )

    return shuffle, batch_size


def _get_seq2seq_config() -> Dict[str, Any]:
    """Get sequence-to-sequence configuration."""
    with st.expander(
        "Configure Sequence-to-Sequence Parameters", expanded=True
    ):
        col1, col2 = st.columns(2)

        with col1:
            prompt_field = st.text_input(
                "Source Field Name",
                help="Field name for source text (prompt) in the dataset",
                key="prompt_field_input",
            )
            max_length = st.number_input(
                "Maximum Sequence Length",
                min_value=1,
                max_value=MAX_SEQ_LENGTH,
                value=DEFAULT_SEQ_LENGTH,
                help="Maximum length of input sequences",
                key="max_length_input",
            )

        with col2:
            response_field = st.text_input(
                "Target Field Name",
                help="Field name for target text (response) in the dataset",
                key="response_field_input",
            )
            truncate = st.checkbox(
                "Truncate Long Sequences",
                value=True,
                help="Whether to truncate sequences longer than max_length",
                key="truncate_checkbox",
            )

    return {
        "prompt_field": prompt_field,
        "response_field": response_field,
        "max_length": max_length,
        "truncate": truncate,
    }


def _show_dataset_preview(data_config: DataConfig) -> None:
    """Show dataset preview if requested."""
    from backend.data_pipeline import create_pipeline

    if st.button("Preview Dataset", type="secondary", key="preview_dataset"):
        try:
            pipeline = create_pipeline(data_config)
            tab1, tab2 = st.tabs(
                ["Raw Data Preview", "Tokenized Output Preview"]
            )
            with tab1:
                _display_raw_preview(pipeline)
            with tab2:
                _display_tokenized_preview(pipeline)
        except Exception as e:
            st.error(f"Error creating preview: {e}")


def _display_raw_preview(pipeline: DataPipeline) -> None:  # type: ignore
    """Display raw data preview."""
    st.markdown("#### Human-Readable Source Data")
    try:
        with st.spinner("Loading raw preview..."):
            df = pipeline.get_preview()
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
        st.error(f"Error loading raw preview: {e}")


def _display_tokenized_preview(pipeline: DataPipeline) -> None:  # type: ignore
    """Display tokenized preview."""
    st.markdown("#### Model Input After Tokenization")
    try:
        with st.spinner("Loading tokenized preview..."):
            tokenized_examples = pipeline.get_preview(tokenized=True)
            with treescope.active_autovisualizer.set_scoped(
                treescope.ArrayAutovisualizer()
            ):
                html = treescope.render_to_html(tokenized_examples)
                components.html(html, height=250, scrolling=True)
            with st.expander("Tokenizer Decoded Text", expanded=True):
                if "input" in tokenized_examples:
                    turns = _extract_conversation_turns(
                        pipeline, tokenized_examples
                    )
                    if turns:
                        df = pd.DataFrame(turns)
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
        st.error(f"Error loading tokenized preview: {e}")


def _extract_conversation_turns(
    pipeline: DataPipeline, tokenized_examples: dict  # type: ignore
) -> List[dict]:
    """Extract conversation turns from tokenized examples."""
    turns: List[dict] = []
    for i in range(len(tokenized_examples["input"])):
        decoded_text = pipeline.tokenizer.decode(tokenized_examples["input"][i])

        if (
            "<start_of_turn>user" in decoded_text
            and "<start_of_turn>model" in decoded_text
        ):
            user_part = decoded_text.split("<start_of_turn>model")[0].strip()
            model_part = (
                "<start_of_turn>model"
                + decoded_text.split("<start_of_turn>model")[1].strip()
            )
            turns.append({"user": user_part, "model": model_part})

    return turns
