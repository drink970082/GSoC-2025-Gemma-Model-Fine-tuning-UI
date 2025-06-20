import os
import time
from pathlib import Path

import streamlit as st

from app.components.training_dashboard.kpi_panel import display_kpis
from app.components.training_dashboard.logs_panel import display_live_logs
from app.components.training_dashboard.plots_panel import display_native_plots
from app.components.training_dashboard.usage_panel import (
    display_system_usage_panel,
)
from backend.manager.global_manager import (
    get_process_manager,
    get_status_manager,
    get_tensorboard_manager,
)
from config.app_config import get_config

config = get_config()


@st.fragment(run_every=1)
def update_tensorboard_data():
    """Update TensorBoard data every second (separate from display fragments)."""
    manager = get_tensorboard_manager()
    # This triggers the update check and loads fresh data if needed
    manager.get_data()


@st.fragment(run_every=1)
def poll_training_status():
    manager = get_process_manager()
    if (
        not manager.is_training_running()
        and st.session_state.session_started_by_app
    ):
        st.session_state.session_started_by_app = False
        st.rerun()


@st.fragment(run_every=1)
def update_status():
    manager = get_status_manager()
    status = manager.get()
    st.info(f"Training in progress... Status: {status}")


def shutdown_training():
    process_manager = get_process_manager()
    tensorboard_manager = get_tensorboard_manager()
    with st.spinner(
        "Shutdown requested. Waiting for processes to terminate...",
        show_time=True,
    ):
        # Clear TensorBoard cache when stopping training
        tensorboard_manager.clear_cache()
        if process_manager.stop_all_processes():
            st.session_state.shutdown_requested = True
        elif process_manager.stop_all_processes(mode="force"):
            st.session_state.shutdown_requested = True
        else:
            st.error(
                "Failed to stop training processes. Please check the logs."
            )
    if st.session_state.shutdown_requested:
        st.session_state.session_started_by_app = False
        st.success(
            "All processes have been shut down. Redirecting to welcome page..."
        )
        st.session_state.view = "welcome"
        time.sleep(2)
        st.rerun()
    else:
        st.error("Failed to stop training processes. Please check the logs.")


def show_training_dashboard_view():
    """Display the training interface."""
    process_manager = get_process_manager()
    tensorboard_manager = get_tensorboard_manager()
    status_manager = get_status_manager()
    st.title("LLM Fine-Tuning Dashboard")
    if "shutdown_requested" not in st.session_state:
        st.session_state.shutdown_requested = False
    if "session_started_by_app" not in st.session_state:
        st.session_state.session_started_by_app = False
    is_currently_training = process_manager.is_training_running()
    print("--------------------------------")
    print("is_currently_training", is_currently_training)
    print("session_was_started_here", st.session_state.session_started_by_app)

    if is_currently_training and not st.session_state.session_started_by_app:
        # State 1: Stale lock file detected
        st.warning(
            "A `training.lock` file was found from a previous session that may have crashed."
        )
        st.info("Please choose how to proceed:")
        col1, col2 = st.columns(2)
        if col1.button(
            "Attempt to Monitor Existing Session", use_container_width=True
        ):
            st.session_state.session_started_by_app = True
            st.rerun()
        if col2.button(
            "Abort Training", type="primary", use_container_width=True
        ):
            shutdown_training()

    elif is_currently_training and st.session_state.session_started_by_app:
        poll_training_status()
        # State 3: Training is actively in progress
        update_status()
        if st.button(
            "Abort Training", type="primary", use_container_width=True
        ):
            shutdown_training()

        st.divider()

        # Dashboard Panels
        update_tensorboard_data()
        display_kpis()
        display_native_plots()
        display_system_usage_panel()
        display_live_logs()

    else:
        # State 4: Not training (initial, completed, or failed state)
        st.session_state.session_started_by_app = False

        final_status = status_manager.get()

        if "Error" in final_status:
            st.error(f"Training Failed: {final_status}")
            st.info(
                "The full error traceback may be available in the logs below."
            )
            display_live_logs(expanded=True)
            if st.button("Reset Application"):
                # Clear TensorBoard cache when resetting
                tensorboard_manager.clear_cache()
                if process_manager.stop_all_processes(mode="force"):
                    st.success("Successfully reset the application.")
                else:
                    st.error(
                        "Failed to reset the application. Please check the logs."
                    )
                st.rerun()
        else:
            if not os.path.exists(config.CHECKPOINT_FOLDER):
                return None

            subdirs = [p for p in Path(config.CHECKPOINT_FOLDER).iterdir()]
            if not subdirs:
                return None

            # Return the path of the most recently created directory
            latest_subdir = max(subdirs, key=lambda p: p.stat().st_ctime)
            if latest_subdir:
                st.success(
                    f"A trained model checkpoint was found ({latest_subdir})."
                )
                if st.button("Go to Inference Playground", type="primary"):
                    st.session_state.view = "inference"
                    st.rerun()

            st.info("No active or completed training runs found.")
            if st.button("Create a New Model to Start Training"):
                st.session_state.view = "create"
                st.rerun()


if __name__ == "__main__":
    show_training_dashboard_view()
