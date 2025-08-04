import streamlit as st


def show_model_name_section():
    """Display the model name input section."""
    model_name = st.text_input(
        "Enter a name for your model",
        placeholder="e.g., gemma-3-1b-LoRA",
        help="Choose a descriptive name for your fine-tuned model",
        key="model_name",
    )
    if not model_name:
        return ""
    return model_name.strip()
