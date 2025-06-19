from backend.core.model import ModelFactory
import streamlit as st
from backend.inferencer import InferenceService
from config.training_config import DEFAULT_MODEL_CONFIG, ModelConfig


def show_inference_panel():
    """Display the inference playground."""
    st.header("Inference Playground")
    st.write("Test your newly trained model!")

    # Initialize inference service
    if "inference_service" not in st.session_state:
        st.session_state.inference_service = InferenceService(
            ModelConfig(**DEFAULT_MODEL_CONFIG)
        )

    service = st.session_state.inference_service

    # Load model if not already loaded
    if not service.is_loaded():
        with st.spinner("Loading trained model..."):
            if not service.load_model():
                st.error(
                    "Failed to load trained model. Please ensure training completed successfully."
                )
                return
        st.success("Model loaded successfully!")

    # Input section
    st.subheader("Input")
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Type your message here...",
        height=100,
    )

    # Generate button
    if st.button("Generate Response", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating response..."):
            try:
                response = service.generate(prompt)

                st.subheader("Response")
                st.write(response)

                # Show token info
                input_tokens = len(service.tokenizer.encode(prompt))
                output_tokens = len(service.tokenizer.encode(response))
                st.caption(
                    f"Input tokens: {input_tokens} | Output tokens: {output_tokens}"
                )

            except Exception as e:
                st.error(f"Error during generation: {str(e)}")


if __name__ == "__main__":
    show_inference_panel()
