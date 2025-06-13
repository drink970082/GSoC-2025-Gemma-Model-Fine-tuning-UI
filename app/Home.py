import json
import time

import plotly.graph_objects as go
import streamlit as st

# from app.components.model_playground import show_model_playground
from app.pages.create_model import show_create_model
from app.pages.training import show_training

# Set page config
st.set_page_config(
    page_title="Gemma Fine-tuning UI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
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

    # Add sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Select a page",
            ["Create Model", "Training", "Model Playground"],
            index=(
                0
                if st.session_state.page == "home"
                else 1 if st.session_state.page == "training" else 2
            ),
            label_visibility="collapsed",
        )

        # Update page state based on selection
        if page == "Create Model":
            st.session_state.page = "home"
        elif page == "Training":
            st.session_state.page = "training"
        else:
            st.session_state.page = "playground"

    # Show content based on selected page
    if st.session_state.page == "home":
        st.header("Create Model")
        show_create_model()
    elif st.session_state.page == "training":
        st.header("Training")
        show_training()
    elif st.session_state.page == "playground":
        st.header("Model Playground")
        if st.session_state.current_model:
            # show_model_playground(st.session_state.current_model)
            st.info("Model Playground feature coming soon!")
        else:
            st.warning(
                "Please create and train a model first to use the playground."
            )


if __name__ == "__main__":
    main()
