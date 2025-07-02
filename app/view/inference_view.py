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

    # Step 1: Ensure the model is loaded and get the service instance.
    service = ensure_model_loaded(training_service)

    # Step 2: If the model is loaded, show the interactive playground.
    if service:
        st.divider()
        show_inference_playground(service)


if __name__ == "__main__":
    show_inference_view()
