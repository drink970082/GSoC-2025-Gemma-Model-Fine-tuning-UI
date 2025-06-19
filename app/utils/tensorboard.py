import streamlit as st

from backend.utils.manager.TensorboardManager import TensorBoardDataManager


@st.fragment(run_every=1)
def update_tensorboard_data():
    """Update TensorBoard data every second (separate from display fragments)."""
    manager = get_tensorboard_manager()
    # This triggers the update check and loads fresh data if needed
    manager.get_data()


def get_tensorboard_manager() -> TensorBoardDataManager:
    """Get or create TensorBoard data manager from session state."""
    if "tensorboard_manager" not in st.session_state:
        st.session_state.tensorboard_manager = TensorBoardDataManager()

    return st.session_state.tensorboard_manager


def reset_tensorboard_training_time():
    """Reset the TensorBoard manager's training start time."""
    if "tensorboard_manager" in st.session_state:
        st.session_state.tensorboard_manager.reset_training_time()


def clear_tensorboard_cache():
    """Clear TensorBoard cache from session state."""
    if "tensorboard_manager" in st.session_state:
        st.session_state.tensorboard_manager.clear_cache()


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
