import streamlit as st
from dataclasses import asdict
from config.dataclass import TrainingConfig
from typing import Optional

def show_configuration_preview(config: Optional[TrainingConfig]) -> None:
    """Display the configuration preview section."""
    if not config:
        st.warning("No configuration available to preview")
        return

    with st.expander("Review Your Configuration", expanded=True):
        st.json(asdict(config))
