import atexit
import json
import os
import time

import plotly.graph_objects as go
import streamlit as st

# from app.components.model_playground import show_model_playground
from app.panel.create_model import show_create_model_panel
from app.panel.training import show_training_panel
from app.panel.inference import show_inference_panel
from backend.inferencer import InferenceService
from backend.utils.manager.GlobalManager import get_process_manager
from backend.utils.monitor import check_training_status
from config.training_config import MODEL_ARTIFACT

# Set page config
st.set_page_config(
    page_title="Gemma Fine-tuning UI",
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


# --- Welcome Modal & View Management ---
def display_welcome_modal(is_training: bool):
    """The central navigation hub that adapts to the application's state."""
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            st.markdown(
                "<h2 style='text-align: center;'>Gemma Fine-Tuning</h2>",
                unsafe_allow_html=True,
            )

            if is_training:
                st.markdown(
                    "<p style='text-align: center;'>An active fine-tuning process is running.</p>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Go to Live Monitoring",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.view = "training"
                    st.rerun()
                if st.button("Abort and Start New", use_container_width=True):
                    st.session_state.abort_confirmation = True
                    st.rerun()
            else:
                st.markdown(
                    "<p style='text-align: center;'>Choose an option to get started.</p>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Start New Fine-Tuning",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.view = "create"
                    st.rerun()
                if st.button(
                    "Inference Existing Model", use_container_width=True
                ):
                    st.session_state.view = "inference"
                    st.rerun()

            if st.session_state.get("abort_confirmation"):
                st.warning(
                    "Are you sure you want to abort the current training process?"
                )
                c1, c2 = st.columns(2)
                if c1.button("Yes, Abort", use_container_width=True):
                    get_process_manager().stop_all_processes(mode="force")
                    st.session_state.abort_confirmation = False
                    st.session_state.view = "create"
                    st.rerun()
                if c2.button("No, Cancel", use_container_width=True):
                    st.session_state.abort_confirmation = False
                    st.rerun()


# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title("Gemma Fine-Tuning UI")

    # Initialize session state for view management
    if "view" not in st.session_state:
        st.session_state.view = "welcome"
    if "abort_confirmation" not in st.session_state:
        st.session_state.abort_confirmation = False

    is_training = check_training_status()
    # If training is detected on load, go directly to the dashboard
    if is_training and st.session_state.view != "training":
        st.session_state.view = "training"
        st.rerun()

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("Back to Home", use_container_width=True):
            st.session_state.view = "welcome"
            st.rerun()

    # # Sidebar for navigation between unlocked stages
    # with st.sidebar:
    #     st.header("Workflow")
    #     if st.button(
    #         "1. Model Creation", use_container_width=True, disabled=is_training
    #     ):
    #         st.session_state.view = "create"
    #         st.rerun()
    #     if st.button("2. Training Dashboard", use_container_width=True):
    #         st.session_state.view = "training"
    #         st.rerun()
    #     if st.button(
    #         "3. Inference Playground",
    #         use_container_width=True,
    #         disabled=is_training,
    #     ):
    #         st.session_state.view = "inference"
    #         st.rerun()

    # Main panel displays the current view
    if st.session_state.view == "welcome":
        display_welcome_modal(is_training)
    elif st.session_state.view == "create":
        show_create_model_panel()
    elif st.session_state.view == "training":
        show_training_panel()
    elif st.session_state.view == "inference":
        show_inference_panel()


if __name__ == "__main__":
    main()
