import streamlit as st

from backend.inferencer import Inferencer
from config.training_config import DEFAULT_MODEL_CONFIG, ModelConfig


def ensure_model_loaded() -> Inferencer | None:
    """
    Initializes the Inferencer and ensures the model is loaded.
    Displays status messages to the user.
    Returns the service instance if loaded, otherwise None.
    """
    if "inference_service" not in st.session_state:
        st.session_state.inferencer = Inferencer(
            ModelConfig(**DEFAULT_MODEL_CONFIG)
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
