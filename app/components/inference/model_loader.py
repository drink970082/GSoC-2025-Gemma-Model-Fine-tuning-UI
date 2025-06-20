import streamlit as st

from backend.inferencer import Inferencer
from backend.manager.global_manager import get_process_manager
from config.app_config import ModelConfig


def ensure_model_loaded() -> Inferencer | None:
    """
    Initializes the Inferencer and ensures the model is loaded.
    Displays status messages to the user.
    Returns the service instance if loaded, otherwise None.
    """
    if "inference_service" not in st.session_state:
        process_manager = get_process_manager()
        st.session_state.inferencer = Inferencer(
            ModelConfig(**process_manager.model_config)
        )

    service = st.session_state.inferencer

    if not service.is_loaded():
        with st.spinner("Loading trained model..."):
            if not service.load_model():
                st.error(
                    "Failed to load trained model. "
                    "Please ensure training completed successfully."
                )
                return None
        st.success("Model loaded successfully!")

    return service
