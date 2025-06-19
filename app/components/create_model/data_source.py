import json
import tempfile

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import treescope

from backend.data_pipeline import create_pipeline


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
                stringio = uploaded_file.getvalue().decode("utf-8")
                lines = stringio.splitlines()
                data = []
                for line in lines:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
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
