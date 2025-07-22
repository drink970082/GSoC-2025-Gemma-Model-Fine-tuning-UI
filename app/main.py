import streamlit as st

from app.view.create_model_view import show_create_model_view
from app.view.inference_view import show_inference_view
from app.view.training_dashboard_view import show_training_dashboard_view
from app.view.welcome_view import show_welcome_modal
from services.di_container import get_service
from config.dataclass import TrainingConfig, DataConfig, ModelConfig


def _initialize_session_state() -> None:
    """Register the session state."""
    if "view" not in st.session_state:
        st.session_state.view = "welcome"
    if "session_started_by_app" not in st.session_state:
        st.session_state.session_started_by_app = False
    if "abort_training" not in st.session_state:
        st.session_state.abort_training = False
    if "inferencer" not in st.session_state:
        st.session_state.inferencer = None
    if "frozen_kpi_data" not in st.session_state:
        st.session_state.frozen_kpi_data = {}
    if "frozen_log" not in st.session_state:
        st.session_state.frozen_log = "No logs available."
    if "frozen_training_metrics" not in st.session_state:
        st.session_state.frozen_training_metrics = {}


def default_config() -> TrainingConfig:
    """Default training configuration for testing."""
    return TrainingConfig(
        model_name="gemma-3-1b-test",
        data_config=DataConfig(
            source="tensorflow",
            dataset_name="mtnt",
            split="train",
            shuffle=True,
            batch_size=4,
            seq2seq_in_prompt="src",
            seq2seq_in_response="dst",
            seq2seq_max_length=200,
            seq2seq_truncate=True,
            config=None,
        ),
        model_config=ModelConfig(
            model_variant="Gemma3_1B",
            epochs=1,
            learning_rate=1e-4,
            method="Standard",
        )
    )

# --- Main Application ---
def main() -> None:
    """Main function to run the Streamlit app."""
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
    training_service = get_service("training_service")
    _initialize_session_state()

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("Back to Home", use_container_width=True):
            st.session_state.view = "welcome"
            st.rerun()

    # Main panel displays the current view
    if st.session_state.view == "welcome":
        show_welcome_modal(training_service)
    elif st.session_state.view == "create_model":
        show_create_model_view(training_service)
    elif st.session_state.view == "training_dashboard":
        show_training_dashboard_view(training_service)
    elif st.session_state.view == "inference":
        show_inference_view()


if __name__ == "__main__":
    main()
