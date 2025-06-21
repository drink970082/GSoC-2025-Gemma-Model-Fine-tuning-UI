import os
import time

import streamlit as st
from pathlib import Path
from typing import Optional

from app.services.global_service import get_training_service
from config.app_config import get_config

config = get_config()


def _find_latest_checkpoint() -> Optional[Path]:
    """Scans the checkpoint folder and returns the path to the latest one."""
    checkpoint_folder = Path(config.CHECKPOINT_FOLDER)
    if not checkpoint_folder.exists() or not checkpoint_folder.is_dir():
        return None

    subdirs = [p for p in checkpoint_folder.iterdir() if p.is_dir()]
    if not subdirs:
        return None

    return max(subdirs, key=lambda p: p.stat().st_ctime)


def _create_shutdown_button(label: str):
    if st.button(label, type="primary", use_container_width=True):
        training_service = get_training_service()
        with st.spinner(
            "Waiting for processes to terminate...", show_time=True
        ):
            shutdown_ok = training_service.stop_training()
            if not shutdown_ok:
                st.info(
                    "Graceful shutdown failed. Attempting forceful shutdown..."
                )
                shutdown_ok = training_service.stop_training(mode="force")

        if shutdown_ok:
            st.success(
                "All processes have been shut down. Redirecting to welcome page..."
            )
            st.session_state.session_started_by_app = False
            st.session_state.view = "welcome"
            time.sleep(2)
            st.rerun()
        else:
            st.error(
                "Failed to stop training processes. Please check the logs."
            )


def _create_next_step_buttons():
    """Displays the primary 'next step' buttons after a run is complete."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Go to Inference Playground",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.view = "inference"
            st.rerun()
    with col2:
        if st.button("Create New Model", use_container_width=True):
            st.session_state.view = "create"
            st.rerun()


def display_control_panel():
    """
    Displays the main status and control panel at the top of the dashboard.
    This panel changes based on the training state (active, failed, completed).
    """
    training_service = get_training_service()
    is_training_active = training_service.is_training_running()
    status = training_service.get_training_status()
    if is_training_active:

        st.info(f"Training in progress... Status: {status}")
        _create_shutdown_button("Abort Training")
    else:
        latest_checkpoint = _find_latest_checkpoint()
        if "Error" in status:
            # State: Training has failed
            st.error(f"Training Failed: {status}")
            _create_shutdown_button("Reset Application")
        elif latest_checkpoint is None:
            st.warning(
                "Training finished but no checkpoint found. Please create a new model."
            )
            _create_shutdown_button("Reset Application")
        else:
            st.success("Training concluded successfully.")
            st.info(f"Latest checkpoint found: {latest_checkpoint.name}")
            _create_next_step_buttons()
