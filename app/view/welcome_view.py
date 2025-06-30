import streamlit as st
from services.training_service import TrainingService


def display_welcome_modal(training_service: TrainingService):
    """The central navigation hub that adapts to the application's state."""
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.container(border=True):
            st.markdown(
                "<h2 style='text-align: center;'>Gemma Fine-Tuning</h2>",
                unsafe_allow_html=True,
            )

            if training_service.is_training_running():
                st.markdown(
                    "<p style='text-align: center;'>An active fine-tuning process is running.</p>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Go to Live Monitoring",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.view = "training_dashboard"
                    st.rerun()
                if st.button("Abort and Start New", use_container_width=True):
                    st.session_state.abort_confirmation = True
                    st.rerun()
            else:
                st.markdown(
                    "<p style='text-align: center;'>Choose an option to get started.</p>",
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Start New Fine-Tuning",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.view = "create_model"
                    st.rerun()
                if st.button(
                    "Inference Existing Model", use_container_width=True
                ):
                    st.session_state.view = "inference"
                    st.rerun()

            if st.session_state.get("abort_confirmation"):
                st.warning(
                    "Are you sure you want to abort the current training process?"
                )
                c1, c2 = st.columns(2)
                if c1.button("Yes, Abort", use_container_width=True):
                    training_service.stop_training(mode="force")
                    st.session_state.abort_confirmation = False
                    st.session_state.view = "create_model"
                    st.rerun()
                if c2.button("No, Cancel", use_container_width=True):
                    st.session_state.abort_confirmation = False
                    st.rerun()
