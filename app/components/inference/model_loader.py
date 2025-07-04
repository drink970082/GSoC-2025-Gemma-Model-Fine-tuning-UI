import streamlit as st
from backend.inferencer import Inferencer
from services.training_service import TrainingService
from pathlib import Path


def ensure_model_loaded(
    training_service: TrainingService, checkpoint_name: str
) -> Inferencer | None:
    """
    Initializes the Inferencer and ensures the model is loaded.
    Displays status messages to the user.
    Returns the service instance if loaded, otherwise None.
    """
    training_config = training_service.get_training_config()
    inferencer = Inferencer(training_config, training_service.work_dir)
    if not inferencer.is_loaded():
        with st.spinner("Loading trained model..."):
            checkpoint_path = str(Path(inferencer.work_dir) / checkpoint_name)
            if not inferencer.load_model(checkpoint_path=checkpoint_path):
                st.error(
                    "Failed to load trained model. "
                    "Please ensure training completed successfully."
                )
                return None
        st.success("Model loaded successfully!")

    return inferencer
