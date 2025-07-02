import streamlit as st

from backend.inferencer import Inferencer
from config.app_config import ModelConfig
from services.training_service import TrainingService


def ensure_model_loaded(training_service: TrainingService) -> Inferencer | None:
    """
    Initializes the Inferencer and ensures the model is loaded.
    Displays status messages to the user.
    Returns the service instance if loaded, otherwise None.
    """
    if "inference_service" not in st.session_state:
        model_config = training_service.get_model_config()
        st.session_state.inferencer = Inferencer(
            ModelConfig(**model_config), training_service.work_dir
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
