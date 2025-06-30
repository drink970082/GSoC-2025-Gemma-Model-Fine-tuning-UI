import os
import streamlit as st
from config.app_config import get_config
from services.training_service import TrainingService

config = get_config()


@st.fragment(run_every=1)
def display_logs_panel(training_service: TrainingService):
    """Display the live training logs panel as a code area with a score."""
    st.subheader("Live Training Logs")
    stdout, stderr = training_service.get_log_contents()
    log_content = stdout or "Waiting for training process to start..."
    error_count = 0
    if stderr:
        log_content += "\n\n--- ERRORS ---\n" + stderr
        error_count = stderr.lower().count("error")

    st.metric("Error Count", error_count)
    st.code(log_content, language="log", height=400)
