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
        try:
            config_dict = asdict(config)
            st.json(config_dict)
        except Exception as e:
            st.error(f"Error displaying configuration: {e}")
