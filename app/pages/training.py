import atexit
import os
import time
from datetime import datetime

import streamlit as st

from app.components.dashboard.kpi_panel import display_kpis
from app.components.dashboard.logs_panel import display_live_logs
from app.components.dashboard.plots_panel import display_native_plots
from app.components.dashboard.usage_panel import display_system_usage_panel
from app.utils.tensorboard import (
    clear_tensorboard_cache,
    display_tensorboard_iframe,
    reset_tensorboard_training_time,
    update_tensorboard_data,
)
from backend.utils.manager.ProcessManager import ProcessManager
from backend.utils.monitor import (
    check_training_status,
    get_training_status,
    initialize_session_state,
)
from config.training_config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    LOCK_FILE,
    MODEL_ARTIFACT,
)


@st.fragment(run_every=1)
def poll_training_status():
    if not check_training_status() and st.session_state.session_started_by_app:
        st.session_state.session_started_by_app = False
        st.rerun()


@st.fragment(run_every=1)
def update_status():
    status = get_training_status()
    st.info(f"Training in progress... Status: {status}")


def show_training():
    """Display the training interface."""
    if "process_manager" not in st.session_state:
        st.session_state.process_manager = ProcessManager()
        atexit.register(st.session_state.process_manager.cleanup)
    st.title("LLM Fine-Tuning Dashboard")
    initialize_session_state()

    is_currently_training = check_training_status()
    session_was_started_here = st.session_state.session_started_by_app
    print("--------------------------------")
    print("is_currently_training", is_currently_training)
    print("session_was_started_here", session_was_started_here)

    if is_currently_training and not session_was_started_here:
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
            "Force Cleanup and Reset",
            type="primary",
            use_container_width=True,
        ):
            if st.session_state.process_manager.stop_all_processes(
                mode="force"
            ):
                st.success(
                    "Successfully cleaned up stale processes. Will reload in 3 seconds."
                )
                time.sleep(3)
            else:
                st.error(
                    "Failed to clean up some processes. Please check the logs."
                )
            st.rerun()

    elif st.session_state.shutdown_requested:
        # State 2: Shutdown in progress
        with st.spinner(
            "Shutdown requested. Waiting for processes to terminate...",
            show_time=True,
        ):
            if not is_currently_training:
                st.session_state.shutdown_requested = False
                st.session_state.session_started_by_app = False
                st.rerun()
            else:
                time.sleep(1)
                st.rerun()

    elif is_currently_training and session_was_started_here:
        poll_training_status()
        # State 3: Training is actively in progress
        update_status()
        # Ensure TensorBoard is running
        if not st.session_state.process_manager.tensorboard_process:
            st.session_state.process_manager.start_tensorboard()

        if st.button(
            "Abort Training", type="primary", use_container_width=True
        ):
            with st.spinner(
                "Shutdown requested. Waiting for processes to terminate...",
                show_time=True,
            ):
                # Clear TensorBoard cache when stopping training
                clear_tensorboard_cache()
                if st.session_state.process_manager.stop_all_processes():
                    st.session_state.shutdown_requested = True
                elif st.session_state.process_manager.stop_all_processes(
                    mode="force"
                ):
                    st.session_state.shutdown_requested = True
            if st.session_state.shutdown_requested:
                st.success("All processes have been shut down.")
                time.sleep(2)
                st.rerun()
            else:
                st.error(
                    "Failed to stop training processes. Please check the logs."
                )

        st.divider()

        # Dashboard Panels
        update_tensorboard_data()
        display_kpis()
        display_native_plots()
        display_system_usage_panel()
        display_live_logs()
        display_tensorboard_iframe()

    else:
        # State 4: Not training (initial, completed, or failed state)
        st.session_state.session_started_by_app = False

        final_status = get_training_status()

        if "Error" in final_status:
            st.error(f"Training Failed: {final_status}")
            st.info(
                "The full error traceback may be available in the logs below."
            )
            display_live_logs(expanded=True)
            if st.button("Reset Application"):
                # Clear TensorBoard cache when resetting
                clear_tensorboard_cache()
                if st.session_state.process_manager.stop_all_processes(
                    mode="force"
                ):
                    st.success("Successfully reset the application.")
                else:
                    st.error(
                        "Failed to reset the application. Please check the logs."
                    )
                st.rerun()
        else:
            if os.path.exists(MODEL_ARTIFACT):
                stats = os.stat(MODEL_ARTIFACT)
                mod_time = datetime.fromtimestamp(stats.st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                st.success(
                    f"A trained model artifact was found (Last Modified: {mod_time})."
                )

            st.header("Start a New Training Run")
            st.info("Click 'Start Fine-Tuning' to begin.")
            if st.button(
                "Start Fine-Tuning", type="primary", use_container_width=True
            ):
                # Reset TensorBoard training time when starting new training
                reset_tensorboard_training_time()
                st.session_state.process_manager.start_training(
                    DEFAULT_DATA_CONFIG,
                    DEFAULT_MODEL_CONFIG,
                )
                st.session_state.session_started_by_app = True
                with st.spinner(
                    "Waiting for training process to initialize..."
                ):
                    start_time = time.time()
                    while (
                        not os.path.exists(LOCK_FILE)
                        and time.time() - start_time < 10
                    ):
                        time.sleep(0.5)
                st.rerun()


if __name__ == "__main__":
    show_training()
