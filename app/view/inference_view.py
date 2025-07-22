import streamlit as st
from app.components.inference.checkpoint_selection import (
    show_checkpoint_selection,
)
from app.components.inference.inference_input_section import (
    show_inference_input_section,
)


def show_inference_view() -> None:
    """Display the inference playground with checkpoint selection and input interface."""
    st.title("Inference Playground")
    st.subheader("Checkpoint Management")
    show_checkpoint_selection()
    st.divider()
    st.subheader("Inference Playground")
    show_inference_input_section()
