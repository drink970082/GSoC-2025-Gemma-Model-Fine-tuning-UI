import streamlit as st

from backend.inferencer import Inferencer


def show_inference_input_section() -> None:
    """Display the inference input section with a prompt input and a generate button."""
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Type your message here...",
        height=100,
    )
    if st.button("Generate Response", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
            return
        _generate_and_display_response(prompt)
        
        
def _generate_and_display_response(prompt: str) -> None:
    """Generate response and display it with token information."""
    if st.session_state.inferencer is None or not st.session_state.inferencer.is_loaded():
        st.error("No model loaded. Please load a checkpoint first.")
        return
    
    with st.spinner("Generating response...", show_time=True):
        try:
            response = st.session_state.inferencer.generate(prompt)
            st.subheader("Response")
            st.write(response)
            _display_token_info(st.session_state.inferencer, prompt, response)
            
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")


def _display_token_info(inferencer: Inferencer, prompt: str, response: str) -> None:
    """Display token information"""
    input_tokens = inferencer.count_tokens(prompt)
    output_tokens = inferencer.count_tokens(response)
    st.caption(f"Input tokens: {input_tokens} | Output tokens: {output_tokens}")       
