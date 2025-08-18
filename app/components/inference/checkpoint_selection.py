from __future__ import annotations

import streamlit as st

from config.app_config import get_config

config = get_config()


def show_checkpoint_selection() -> None:
    if st.session_state.inferencer is None:
        from backend.inferencer import Inferencer

        st.session_state.inferencer = Inferencer(
            work_dir=config.CHECKPOINT_FOLDER
        )
    checkpoints = st.session_state.inferencer.list_checkpoints()
    if not checkpoints:
        st.warning("No checkpoints found.")
        return
    selected_checkpoint = st.selectbox(
        "Select a checkpoint for inference:",
        checkpoints,
        index=0,
        key="checkpoint_selection",
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button(
            "Load checkpoint",
            type="primary",
            use_container_width=True,
            key="load_checkpoint",
        ):
            _load_selected_checkpoint(
                st.session_state.inferencer, selected_checkpoint
            )

    with col2:
        if st.button(
            "Delete",
            type="secondary",
            use_container_width=True,
            key="delete_checkpoint",
        ):
            _delete_selected_checkpoint(
                st.session_state.inferencer, selected_checkpoint
            )


def _load_selected_checkpoint(
    inferencer: Inferencer, selected_checkpoint: str  # type: ignore
) -> None:
    """Load the selected checkpoint."""
    with st.spinner(f"Loading checkpoint: {selected_checkpoint}"):
        if not inferencer.load_model(selected_checkpoint):
            st.error(f"Failed to load checkpoint: {selected_checkpoint}")
            return
        st.success(f"Loaded checkpoint: {selected_checkpoint}")


def _delete_selected_checkpoint(
    inferencer: Inferencer, selected_checkpoint: str  # type: ignore
) -> None:
    """Delete the selected checkpoint."""
    if inferencer.delete_checkpoint(selected_checkpoint):
        st.success(f"Deleted checkpoint: {selected_checkpoint}")
        if inferencer.is_loaded():
            inferencer.clear_model()
        st.rerun()
    else:
        st.error("Failed to delete checkpoint.")
