# app/train_app.py

import atexit
import glob
import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import psutil
import pynvml
import streamlit as st
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

from config.training_config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_MODEL_CONFIG,
    LOCK_FILE,
    MODEL_ARTIFACT,
    STATUS_LOG,
    TRAINER_STDERR_LOG,
    TRAINER_STDOUT_LOG,
    TENSORBOARD_LOGDIR,
    TENSORBOARD_PORT,
)


class ProcessManager:
    """A simple class to manage background subprocesses."""

    # This class remains unchanged
    def __init__(self):
        self.training_process: Optional[subprocess.Popen] = None
        self.tensorboard_process: Optional[subprocess.Popen] = None
        print("ProcessManager initialized.")

    def start_training(self):
        if os.path.exists(LOCK_FILE):
            st.warning("Training is already in progress.")
            return
        st.info("Fine-tuning has started. This may take a while.")
        command = [
            "python",
            "backend/trainer.py",
            "--data_config",
            str(DEFAULT_DATA_CONFIG),
            "--model_config",
            str(DEFAULT_MODEL_CONFIG),
        ]
        trainer_stdout = open(TRAINER_STDOUT_LOG, "w")
        trainer_stderr = open(TRAINER_STDERR_LOG, "w")
        self.training_process = subprocess.Popen(
            command, stdout=trainer_stdout, stderr=trainer_stderr
        )

    def start_tensorboard(self):
        if self.tensorboard_process and self.tensorboard_process.poll() is None:
            return
        command = [
            "tensorboard",
            "--logdir",
            TENSORBOARD_LOGDIR,
            "--port",
            str(TENSORBOARD_PORT),
        ]
        self.tensorboard_process = subprocess.Popen(command)
        time.sleep(5)

    def _terminate_process(
        self, process: Optional[subprocess.Popen], name: str
    ):
        if process and process.poll() is None:
            print(f"Terminating {name} process (PID: {process.pid})...")
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
                print(f"{name} process terminated gracefully.")
            except subprocess.TimeoutExpired:
                print(f"{name} did not terminate gracefully. Forcing kill.")
                process.kill()

    def stop_all_processes(self):
        self._terminate_process(self.training_process, "Training")
        self._terminate_process(self.tensorboard_process, "TensorBoard")

    def cleanup(self):
        print("ATEIXT: Shutting down background processes...")
        self.stop_all_processes()


if "process_manager" not in st.session_state:
    st.session_state.process_manager = ProcessManager()
atexit.register(st.session_state.process_manager.cleanup)


def check_training_status():
    return os.path.exists(LOCK_FILE)


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = {}
    if "shutdown_requested" not in st.session_state:
        st.session_state.shutdown_requested = False
    if "session_started_by_app" not in st.session_state:
        st.session_state.session_started_by_app = False


@st.cache_data(ttl=5)
def load_event_data(log_dir: str) -> Dict[str, pd.DataFrame]:
    # This function remains unchanged
    data = {}
    try:
        event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        if not event_files:
            return data
        latest_event_file = max(event_files, key=os.path.getmtime)
        event_acc = EventAccumulator(
            latest_event_file, size_guidance={"scalars": 0}
        )
        event_acc.Reload()
        for tag in event_acc.Tags()["scalars"]:
            events = event_acc.Scalars(tag)
            data[tag] = pd.DataFrame(
                [(e.wall_time, e.step, e.value) for e in events],
                columns=["wall_time", "step", "value"],
            )
    except Exception as e:
        print(f"Could not read event file: {e}")
    return data


def display_kpis(event_data: Dict[str, pd.DataFrame]):
    # This function remains unchanged
    st.subheader("Panel 1: Key Performance Indicators")
    kpi_cols = st.columns(4)
    model_config = DEFAULT_MODEL_CONFIG
    total_steps = int(model_config.get("epochs", 0))
    latest_step, latest_loss, throughput, eta_str = 0, 0.0, 0.0, "N/A"
    if "loss" in event_data and not event_data["loss"].empty:
        loss_df = event_data["loss"]
        latest_event = loss_df.iloc[-1]
        latest_step, latest_loss = latest_event["step"], latest_event["value"]
        if len(loss_df) > 1:
            last_step_time = (
                loss_df.iloc[-1]["wall_time"] - loss_df.iloc[-2]["wall_time"]
            )
            if last_step_time > 0:
                throughput = 1 / last_step_time
            avg_step_time = (
                loss_df.iloc[-1]["wall_time"] - loss_df.iloc[0]["wall_time"]
            ) / len(loss_df)
            remaining_steps = total_steps - latest_step
            if remaining_steps > 0 and avg_step_time > 0:
                eta_seconds = remaining_steps * avg_step_time
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
    kpi_cols[0].metric("Global Step", f"{latest_step}/{total_steps}")
    kpi_cols[1].metric("Current Loss", f"{latest_loss:.4f}")
    kpi_cols[2].metric("Throughput", f"{throughput:.2f} steps/sec")
    kpi_cols[3].metric("ETA", eta_str)


def display_native_plots(event_data: Dict[str, pd.DataFrame]):
    # This function remains unchanged
    st.subheader("Panel 2: Core Performance Plots")
    if not event_data:
        st.info("Waiting for first metric to be logged...")
        return
    for metric_name, metric_df in event_data.items():
        st.markdown(f"**Metric:** `{metric_name}`")
        st.line_chart(metric_df, x="step", y="value", use_container_width=True)


def display_system_usage_panel():
    # This function remains unchanged
    st.subheader("Panel 3: System Resource Usage")
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_util, gpu_mem, gpu_temp = [], [], []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_util.append(util_info.gpu)
            gpu_mem.append(mem_info.used / (1024**3))
            gpu_temp.append(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        cpu_util_val = psutil.cpu_percent()
        if "history" not in st.session_state or not st.session_state.history:
            st.session_state.history = {
                "gpu_util": [],
                "gpu_mem": [],
                "gpu_temp": [],
                "cpu_util": [],
            }
        history = st.session_state.history
        history["gpu_util"].append(
            sum(gpu_util) / len(gpu_util) if gpu_util else 0
        )
        history["gpu_mem"].append(sum(gpu_mem) / len(gpu_mem) if gpu_mem else 0)
        history["gpu_temp"].append(
            sum(gpu_temp) / len(gpu_temp) if gpu_temp else 0
        )
        history["cpu_util"].append(cpu_util_val)
        for key in history:
            history[key] = history[key][-100:]
        if len(history["cpu_util"]) > 1:
            chart_cols = st.columns(4)
            chart_cols[0].line_chart(
                pd.DataFrame(history["gpu_util"], columns=["Usage (%)"]),
                use_container_width=True,
            )
            chart_cols[0].caption("GPU Utilization (%)")
            chart_cols[1].line_chart(
                pd.DataFrame(history["gpu_mem"], columns=["Memory (GB)"]),
                use_container_width=True,
            )
            chart_cols[1].caption("GPU Memory Usage (GB)")
            chart_cols[2].line_chart(
                pd.DataFrame(history["gpu_temp"], columns=["Temp (°C)"]),
                use_container_width=True,
            )
            chart_cols[2].caption("GPU Temperature (°C)")
            chart_cols[3].line_chart(
                pd.DataFrame(history["cpu_util"], columns=["Usage (%)"]),
                use_container_width=True,
            )
            chart_cols[3].caption("CPU Utilization (%)")
        else:
            st.info("Collecting system usage data...")
    except pynvml.NVMLError:
        st.warning("NVIDIA GPU not detected.")
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def display_live_logs(expanded=False):
    # This function remains unchanged
    st.subheader("Panel 4: Live Training Logs")
    with st.expander("Show/Hide Logs", expanded=expanded):
        log_content = ""
        try:
            with open(TRAINER_STDOUT_LOG, "r") as f:
                log_content += f.read()
        except FileNotFoundError:
            log_content += "Waiting for training process to start..."
        if os.path.exists(TRAINER_STDERR_LOG):
            with open(TRAINER_STDERR_LOG, "r") as f:
                error_content = f.read()
            if error_content:
                log_content += "\n--- ERRORS ---\n" + error_content
        st.code(log_content, language="log")


def manual_cleanup():
    # This function remains unchanged
    st.warning("Attempting to forcefully clean up all processes and files...")
    try:
        subprocess.run(["pkill", "-f", "backend/trainer.py"], check=False)
        subprocess.run(["pkill", "-f", "tensorboard"], check=False)
        st.info(
            "Sent termination signals to 'backend/trainer.py' and 'tensorboard' processes."
        )
    except FileNotFoundError:
        st.error(
            "'pkill' command not found. For Windows, please use Task Manager."
        )
    files_to_delete = [
        LOCK_FILE,
        STATUS_LOG,
        TRAINER_STDOUT_LOG,
        TRAINER_STDERR_LOG,
    ]
    for f in files_to_delete:
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError as e:
            st.error(f"Could not remove file {f}: {e}")
    st.success("Cleanup complete. The application will now reload.")
    time.sleep(3)


# --- Main App Logic ---
st.title("LLM Fine-Tuning Dashboard")
initialize_session_state()

is_currently_training = check_training_status()
session_was_started_here = st.session_state.session_started_by_app

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
        "Force Cleanup and Reset", type="secondary", use_container_width=True
    ):
        manual_cleanup()
        st.rerun()

elif st.session_state.shutdown_requested:
    # State 2: Shutdown in progress
    st.warning("Shutdown requested. Waiting for processes to terminate...")
    if not is_currently_training:
        st.session_state.shutdown_requested = False
        st.session_state.session_started_by_app = False
        st.success("All processes have been shut down.")
        time.sleep(2)
        st.rerun()
    else:
        time.sleep(1)
        st.rerun()

elif is_currently_training and session_was_started_here:
    # State 3: Training is actively in progress

    # --- Display Status and Abort Button ---
    st.info(
        f"Training in progress... Status: {open(STATUS_LOG).read().strip() if os.path.exists(STATUS_LOG) else 'Initializing'}"
    )
    if st.button("Abort Training", type="secondary", use_container_width=True):
        st.session_state.process_manager.stop_all_processes()
        st.session_state.shutdown_requested = True
        st.rerun()
    st.divider()

    # --- Dashboard Panels ---
    event_data = load_event_data(TENSORBOARD_LOGDIR)

    # FIX: Hoisted the panel calls out of the conditional to prevent duplication
    display_kpis(event_data)
    display_native_plots(event_data)
    display_system_usage_panel()
    # Show logs expanded by default only during initialization
    display_live_logs(expanded=(not event_data))

    # RE-ADDED: Automatic rerun loop for a live dashboard experience
    time.sleep(2)
    st.rerun()

else:
    # State 4: Not training (initial, completed, or failed state)
    st.session_state.session_started_by_app = False

    final_status = ""
    if os.path.exists(STATUS_LOG):
        with open(STATUS_LOG, "r") as f:
            final_status = f.read().strip()

    if "Error" in final_status:
        st.error(f"Training Failed: {final_status}")
        st.info("The full error traceback may be available in the logs below.")
        display_live_logs(expanded=True)
        if st.button("Reset Application"):
            manual_cleanup()
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
            st.session_state.process_manager.start_training()
            st.session_state.session_started_by_app = True
            with st.spinner("Waiting for training process to initialize..."):
                start_time = time.time()
                while (
                    not os.path.exists(LOCK_FILE)
                    and time.time() - start_time < 10
                ):
                    time.sleep(0.5)
            st.rerun()
