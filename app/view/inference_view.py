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
    st.header("Inference Playground")
    if "sampler" not in st.session_state:
        st.session_state.sampler = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    show_checkpoint_selection()
    st.divider()
    show_inference_input_section()


if __name__ == "__main__":
    show_inference_view()
