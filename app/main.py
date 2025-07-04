import plotly.graph_objects as go
import streamlit as st

# from app.components.model_playground import show_model_playground
from app.view.welcome_view import display_welcome_modal
from app.view.create_model_view import show_create_model_view
from app.view.inference_view import show_inference_view
from app.view.training_dashboard_view import show_training_dashboard_view
from services.di_container import get_service

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



# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    # register services
    training_service = get_service("training_service")

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
        display_welcome_modal(training_service)
    elif st.session_state.view == "create_model":
        show_create_model_view(training_service)
    elif st.session_state.view == "training_dashboard":
        show_training_dashboard_view(training_service)
    elif st.session_state.view == "inference":
        show_inference_view(training_service)


if __name__ == "__main__":
    main()
