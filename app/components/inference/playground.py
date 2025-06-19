import streamlit as st

from backend.inferencer import Inferencer


def show_inference_playground(service: Inferencer):
    """
    Displays the interactive inference playground components, including
    the prompt input, generate button, and response display.
    """
    st.subheader("Input")
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Type your message here...",
        height=100,
    )

    if st.button("Generate Response", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating response..."):
            try:
                response = service.generate(prompt)

                st.subheader("Response")
                st.write(response)

                # Show token info if the tokenizer is available
                if hasattr(service, "tokenizer") and service.tokenizer:
                    input_tokens = len(service.tokenizer.encode(prompt))
                    output_tokens = len(service.tokenizer.encode(response))
                    st.caption(
                        f"Input tokens: {input_tokens} | Output tokens: {output_tokens}"
                    )

            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
