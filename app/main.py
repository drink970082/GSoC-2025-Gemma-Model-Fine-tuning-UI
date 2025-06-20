import atexit
import json
import os
import time

import plotly.graph_objects as go
import streamlit as st

# from app.components.model_playground import show_model_playground
from app.view.create_model_view import show_create_model_view
from app.view.inference_view import show_inference_view
from app.view.training_dashboard_view import show_training_dashboard_view
from backend.manager.global_manager import get_process_manager

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


def initialize_session_state():
    """Initialize all session state variables."""
    # App view state
    if "view" not in st.session_state:
        st.session_state.view = "welcome"

    # Training state
    if "training_started" not in st.session_state:
        st.session_state.training_started = False

    # Model and data configurations
    if "model_config" not in st.session_state:
        st.session_state.model_config = {}
    if "data_config" not in st.session_state:
        st.session_state.data_config = {}


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
                    st.session_state.view = "training_dashboard"
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
                    st.session_state.view = "create_model"
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
                    st.session_state.view = "create_model"
                    st.rerun()
                if c2.button("No, Cancel", use_container_width=True):
                    st.session_state.abort_confirmation = False
                    st.rerun()


# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title("Gemma Fine-Tuning UI")

    initialize_session_state()

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("Back to Home", use_container_width=True):
            st.session_state.view = "welcome"
            st.rerun()

    # Main panel displays the current view
    if st.session_state.view == "welcome":
        display_welcome_modal(get_process_manager().is_training_running())
    elif st.session_state.view == "create_model":
        show_create_model_view()
    elif st.session_state.view == "training_dashboard":
        show_training_dashboard_view()
    elif st.session_state.view == "inference":
        show_inference_view()


if __name__ == "__main__":
    main()
