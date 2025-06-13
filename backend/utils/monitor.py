import os
import time

import streamlit as st

from config.training_config import LOCK_FILE, STATUS_LOG


def check_training_status():
    """Check if training is currently in progress."""
    return os.path.exists(LOCK_FILE)


def initialize_session_state():
    """Initialize the session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = {}
    if "shutdown_requested" not in st.session_state:
        st.session_state.shutdown_requested = False
    if "session_started_by_app" not in st.session_state:
        st.session_state.session_started_by_app = False


def get_training_status():
    """Get the current training status."""
    if os.path.exists(STATUS_LOG):
        with open(STATUS_LOG, "r") as f:
            return f.read().strip()
    return "Initializing"
