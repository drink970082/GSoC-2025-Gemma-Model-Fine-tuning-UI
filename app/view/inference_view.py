import streamlit as st

from app.components.inference.model_loader import ensure_model_loaded
from app.components.inference.playground import show_inference_playground


def show_inference_view(training_service):
    """
    Assembles and displays the full Inference Playground view by combining
    the model loader and the interactive playground components.
    """
    st.header("Inference Playground")
    st.write("Test your newly trained model!")
    st.divider()
    show_inference_playground()


if __name__ == "__main__":
    show_inference_view()
