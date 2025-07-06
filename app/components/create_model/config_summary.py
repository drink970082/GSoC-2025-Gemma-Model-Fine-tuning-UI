import streamlit as st
from dataclasses import asdict
from config.dataclass import TrainingConfig


def show_configuration_preview(config: dict):
    """Display the configuration preview section."""
    if not config:
        return

    with st.expander("Review Your Configuration", expanded=True):
        st.json(asdict(config))
