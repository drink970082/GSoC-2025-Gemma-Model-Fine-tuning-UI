import json
import time

import plotly.graph_objects as go
import streamlit as st
from components.create_model import show_create_model
from components.import_model import show_import_model
from components.model_playground import show_model_playground
from components.training import show_training

# Set page config
st.set_page_config(
    page_title="Gemma Fine-tuning UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": "https://github.com/drink970082/GSoC-2025-Gemma-Model-Fine-tuning-UI",
        "Report a bug": "https://github.com/drink970082/GSoC-2025-Gemma-Model-Fine-tuning-UI/issues",
        "About": "# Gemma Fine-tuning UI\n A user-friendly interface for fine-tuning Google's Gemma models",
    },
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "training_data" not in st.session_state:
    st.session_state.training_data = {
        "loss": [],
        "learning_rate": [],
        "gpu_memory": [],
    }
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "training_config" not in st.session_state:
    st.session_state.training_config = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None


def main():
    """Main function to run the Streamlit app."""
    st.title("Gemma Fine-Tuning UI")

    # First Sector: Create Model
    st.header("Create Model")
    show_create_model()

    # Second Sector: Training
    if st.session_state.training_active:
        st.header("Training")
        show_training()

    # Third Sector: Results
    if st.session_state.current_model:
        st.header("Model Results")
        show_model_playground(st.session_state.current_model)


if __name__ == "__main__":
    main()
