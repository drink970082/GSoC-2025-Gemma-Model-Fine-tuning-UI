import json
import os
import shutil
import signal
import subprocess
import time
from dataclasses import asdict
from typing import Literal, Optional, Tuple, Any, TextIO
import streamlit as st

from backend.manager.base_manager import BaseManager
from backend.manager.training_state_manager import TrainingStateManager
from config.app_config import get_config, TrainingStatus
from config.dataclass import ModelConfig, TrainingConfig

config = get_config()

MODEL_CONFIG_FILE = "model_config.json"


class ProcessManager(BaseManager):
    """A class to manage background subprocesses and cleanup operations."""

    def __init__(self, training_state_manager: TrainingStateManager) -> None:
        """Initialize the ProcessManager."""
        super().__init__()
        self.training_process: Optional[subprocess.Popen] = None
        self.training_config: Optional[TrainingConfig] = None
        self.stdout_log_path: Optional[str] = None
        self.stderr_log_path: Optional[str] = None
        self._log_stdout_handle: Optional[TextIO] = None
        self._log_stderr_handle: Optional[TextIO] = None
        self.training_state_manager: TrainingStateManager = (
            training_state_manager
        )

    def cleanup(self) -> None:
        """Cleanup method called by atexit."""
        self.terminate_process(mode="force", delete_checkpoint=True)

    def update_config(self, training_config: TrainingConfig) -> None:
        """Update the training configuration."""
        self.training_config = training_config

    def start_training(self) -> None:
        """Starts the training process in a subprocess."""
        if not self.work_dir:
            raise ValueError("Work directory is not set.")

        if (
            self.training_state_manager.get_state().get("status")
            == TrainingStatus.RUNNING.value
        ):
            st.warning("Training is already in progress.")
            return

        if (
            not self.training_config
            or not self.training_config.data_config
            or not self.training_config.model_config
        ):
            raise ValueError("Data or model configuration is not set.")

        command = [
            "python",
            config.TRAINER_MAIN_PATH,
            "--config",
            json.dumps(asdict(self.training_config)),
            "--work_dir",
            self.work_dir,
        ]

        try:
            f_out, f_err = self._open_log_files()
            self.training_process = subprocess.Popen(
                command, stdout=f_out, stderr=f_err
            )
            start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.training_state_manager.mark_running(
                self.training_process.pid, start_time
            )
            self._write_model_config(self.training_config.model_config)
        except Exception as e:
            st.error(f"Failed to start the training subprocess: {e}")
            self.training_state_manager.mark_failed(
                str(e), time.strftime("%Y-%m-%dT%H:%M:%S")
            )
            return

        time.sleep(2)
        if self.training_process.poll() is not None:
            error_output = self.read_stderr_log()
            if error_output:
                st.error("The training process failed to start. Error:")
                st.code(error_output, language="bash")
            else:
                st.error(
                    "The training process failed to start with no error message."
                )
            self._remove_work_dir()
            self.reset_state()
            self.training_state_manager.mark_failed(
                error_output or "Unknown error",
                time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            return

    def terminate_process(
        self,
        mode: Literal["graceful", "force"] = "graceful",
        delete_checkpoint: bool = False,
    ) -> bool:
        """Terminate a process either gracefully or forcefully."""
        terminated = True
        if self.training_process and self.training_process.poll() is None:
            if mode == "graceful":
                self.training_process.send_signal(signal.SIGINT)
                try:
                    self.training_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    return self.terminate_process(
                        mode="force", delete_checkpoint=delete_checkpoint
                    )
            else:
                try:
                    self.training_process.kill()
                    self.training_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    terminated = False
        if terminated:
            if delete_checkpoint:
                self._remove_work_dir()
            self.reset_state()
            self.training_state_manager.mark_idle()
        else:
            self.training_state_manager.mark_orphaned(
                "Process termination failed"
            )
        return terminated

    def force_cleanup(self) -> bool:
        """Stop all managed processes and clean up files."""
        try:
            subprocess.run(
                ["pkill", "-f", config.TRAINER_MAIN_PATH], check=False
            )
            self.training_state_manager.mark_idle()
            self.reset_state()
            return True
        except FileNotFoundError:
            print(
                "'pkill' command not found. Skipping orphaned process cleanup."
            )
            self.training_state_manager.mark_idle()
            self.reset_state()
            return True
        except Exception as e:
            print(f"Error stopping processes: {e}")
            return False

    def read_stdout_log(self) -> str:
        """Reads the content of the standard output log file."""
        if not self.stdout_log_path:
            return ""
        try:
            with open(self.stdout_log_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def read_stderr_log(self) -> str:
        """Reads the content of the standard error log file."""
        if not self.stderr_log_path:
            return ""
        try:
            with open(self.stderr_log_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def get_status(self) -> str:
        """Investigates the lock file and OS to give a definitive status."""
        state = self.training_state_manager.get_state()
        status = state.get("status", TrainingStatus.IDLE.value)
        if status == TrainingStatus.RUNNING.value:
            if self.training_process:
                if self.training_process.poll() is not None:
                    return self._handle_dead_process()
        return status

    def reset_state(self, delete_checkpoint: bool = False) -> None:
        """Performs a full cleanup of files and resets the manager's internal state to idle."""
        self._close_log_files()
        if delete_checkpoint:
            self._remove_work_dir()
        self.training_process = None
        self.app_config = None
        self.work_dir = None
        self.training_state_manager.cleanup()

    def set_work_dir(self, work_dir: Optional[str]) -> None:
        """Set the work directory and derive file paths."""
        super().set_work_dir(work_dir)
        if work_dir:
            self.stdout_log_path = os.path.join(
                work_dir, config.TRAINER_STDOUT_LOG
            )
            self.stderr_log_path = os.path.join(
                work_dir, config.TRAINER_STDERR_LOG
            )
        else:
            self.stdout_log_path = None
            self.stderr_log_path = None

    def _remove_work_dir(self) -> None:
        """Removes the work directory."""
        if self.work_dir:
            shutil.rmtree(self.work_dir)
            self.work_dir = None

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running."""
        if pid is None or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    def _open_log_files(self) -> Tuple[TextIO, TextIO]:
        """Opens training log files and returns the file handles."""
        try:
            self._log_stdout_handle = open(self.stdout_log_path, "w")
            self._log_stderr_handle = open(self.stderr_log_path, "w")
            return self._log_stdout_handle, self._log_stderr_handle
        except OSError as e:
            print(f"Error opening log files: {e}")
            if self._log_stdout_handle:
                self._log_stdout_handle.close()
            raise e

    def _close_log_files(self) -> None:
        """Closes any open log file handles."""
        if self._log_stdout_handle and not self._log_stdout_handle.closed:
            self._log_stdout_handle.close()
            self._log_stdout_handle = None
        if self._log_stderr_handle and not self._log_stderr_handle.closed:
            self._log_stderr_handle.close()
            self._log_stderr_handle = None

    def _write_model_config(self, model_config: ModelConfig) -> None:
        """Writes the model config to the model_config.json file."""
        with open(f"{self.work_dir}/{MODEL_CONFIG_FILE}", "w") as f:
            json.dump(asdict(model_config), f)

    def _handle_dead_process(self) -> str:
        """Handle case where process is dead but state says RUNNING."""
        if self.training_process and self.training_process.poll() is not None:
            exit_code = self.training_process.returncode
            if exit_code == 0:
                self.training_state_manager.mark_finished(
                    time.strftime("%Y-%m-%dT%H:%M:%S")
                )
                
                return TrainingStatus.FINISHED.value
            else:
                error_msg = (
                    self.read_stderr_log()
                    or "Process exited with non-zero code"
                )
                self.training_state_manager.mark_failed(
                    error_msg, time.strftime("%Y-%m-%dT%H:%M:%S")
                )
                return TrainingStatus.FAILED.value
        else:
            self.training_state_manager.mark_orphaned("Process not found")
            return TrainingStatus.ORPHANED.value
