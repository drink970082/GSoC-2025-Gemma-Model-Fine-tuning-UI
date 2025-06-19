import os

import streamlit as st

from config.training_config import TRAINER_STDERR_LOG, TRAINER_STDOUT_LOG


@st.fragment(run_every=1)
def display_live_logs():
    """Display the live training logs panel as a code area with a score."""
    st.subheader("Live Training Logs")
    log_content = ""
    error_count = 0
    try:
        with open(TRAINER_STDOUT_LOG, "r") as f:
            log_content += f.read()
    except FileNotFoundError:
        log_content += "Waiting for training process to start..."

    if os.path.exists(TRAINER_STDERR_LOG):
        with open(TRAINER_STDERR_LOG, "r") as f:
            error_content = f.read()
        if error_content:
            log_content += "\n--- ERRORS ---\n" + error_content
            error_count += error_content.lower().count("error")

    # Score: count of 'error' lines in logs
    error_score = log_content.lower().count("error")
    st.metric("Error Count", error_score)
    st.code(log_content, language="log", height=400)
