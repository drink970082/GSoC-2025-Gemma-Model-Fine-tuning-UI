import streamlit as st
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


# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    # register services
    training_service = get_service("training_service")

    st.title("Gemma Fine-Tuning UI")
    if "view" not in st.session_state:
        st.session_state.view = "welcome"

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
        show_inference_view()


if __name__ == "__main__":
    main()
