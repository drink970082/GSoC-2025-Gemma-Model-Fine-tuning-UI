import streamlit as st


def show_model_name_section():
    """Display the model name input section."""
    return st.text_input(
        "Enter a name for your model",
        placeholder="e.g., gemma-2b-chat-finetuned",
        value="gemma-3-1b-LoRA",
    )
