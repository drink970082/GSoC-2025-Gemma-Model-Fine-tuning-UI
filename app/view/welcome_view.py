import streamlit as st
from services.training_service import TrainingService


def show_welcome_modal(training_service: TrainingService) -> None:
    """The central navigation hub that adapts to the application's state."""
    st.title("Gemma Fine-Tuning")
    
    if training_service.is_training_running() == "RUNNING":
        _show_running_training_interface()
    else:
        _show_main_navigation_interface()
    
    _show_abort_confirmation_dialog(training_service)


def _show_running_training_interface() -> None:
    """Show interface when training is running."""
    st.info("An active fine-tuning process is running.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Go to Live Monitoring", use_container_width=True, type="primary"):
            st.session_state.view = "training_dashboard"
            st.rerun()
    with col2:
        if st.button("Abort and Start New", use_container_width=True):
            st.session_state.abort_confirmation = True
            st.rerun()


def _show_main_navigation_interface() -> None:
    """Show main navigation interface when no training is running."""
    st.info("Choose an option to get started.")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start New Fine-Tuning", use_container_width=True, type="primary"):
            st.session_state.view = "create_model"
            st.rerun()
    with col2:
        if st.button("Inference Existing Model", use_container_width=True):
            st.session_state.view = "inference"
            st.rerun()


def _show_abort_confirmation_dialog(training_service: TrainingService) -> None:
    """Show abort confirmation dialog if requested."""
    if not st.session_state.get("abort_confirmation"):
        return
    
    st.warning("Are you sure you want to abort the current training process?")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Yes, Abort", use_container_width=True):
            training_service.stop_training(mode="force")
            st.session_state.abort_confirmation = False
            st.session_state.view = "create_model"
            st.rerun()
    
    with col2:
        if st.button("No, Cancel", use_container_width=True):
            st.session_state.abort_confirmation = False
            st.rerun()