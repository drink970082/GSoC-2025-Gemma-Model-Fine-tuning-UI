import streamlit as st


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
