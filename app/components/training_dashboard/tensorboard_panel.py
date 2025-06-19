import streamlit as st


def display_tensorboard_iframe(port: int = 6007) -> None:
    """Display TensorBoard directly in the Streamlit app using an iframe."""
    st.markdown(
        f"""
        <iframe
            src="http://localhost:{port}"
            width="100%"
            height="800"
            frameborder="0"
            allowfullscreen
        ></iframe>
        """,
        unsafe_allow_html=True,
    )
