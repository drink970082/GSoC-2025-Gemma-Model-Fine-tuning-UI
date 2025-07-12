import streamlit as st
from app.components.inference.checkpoint_selection import (
    show_checkpoint_selection,
)
from app.components.inference.inference_input_section import (
    show_inference_input_section,
)


def show_inference_view():
    """
    Displays the interactive inference playground components, including
    the prompt input, generate button, and response display.
    """
    st.title("Inference Playground")
    show_checkpoint_selection()
    st.divider()
    show_inference_input_section()


if __name__ == "__main__":
    show_inference_view()
