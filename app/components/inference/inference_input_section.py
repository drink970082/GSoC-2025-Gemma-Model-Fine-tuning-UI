import streamlit as st


def show_inference_input_section():
    st.subheader("Inference Playground")

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
                if (
                    "sampler" not in st.session_state
                    or st.session_state.sampler is None
                ):
                    st.error(
                        "No sampler found. Please load a checkpoint first."
                    )
                    return
                response = st.session_state.sampler.chat(prompt)

                st.subheader("Response")
                st.write(response)

                # Show token info if the tokenizer is available
                if st.session_state.tokenizer:
                    input_tokens = len(
                        st.session_state.tokenizer.encode(prompt)
                    )
                    output_tokens = len(
                        st.session_state.tokenizer.encode(response)
                    )
                if st.session_state.tokenizer:
                    input_tokens = len(
                        st.session_state.tokenizer.encode(prompt)
                    )
                    output_tokens = len(
                        st.session_state.tokenizer.encode(response)
                    )
                    st.caption(
                        f"Input tokens: {input_tokens} | Output tokens: {output_tokens}"
                    )

            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
