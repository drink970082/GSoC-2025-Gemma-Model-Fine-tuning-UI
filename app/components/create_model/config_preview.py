import streamlit as st


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
