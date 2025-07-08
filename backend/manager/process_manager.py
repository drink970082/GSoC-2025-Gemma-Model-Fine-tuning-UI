import os
import signal
import subprocess
import time
from typing import Literal, Optional
from dataclasses import asdict
import json
import streamlit as st
import shutil
from backend.manager.base_manager import BaseManager
from config.app_config import TrainingStatus, get_config
from config.dataclass import TrainingConfig, ModelConfig

config = get_config()


class ProcessManager(BaseManager):
    """A class to manage background subprocesses and cleanup operations."""

    def __init__(self):
        super().__init__()
        self.training_process: Optional[subprocess.Popen] = None
        self.training_config: Optional[TrainingConfig] = None
        # Paths for managed files
        self.lock_file_path = config.LOCK_FILE
        self.stdout_log_path = None
        self.stderr_log_path = None
        # File handles
        self._log_stdout_handle = None
        self._log_stderr_handle = None

    def cleanup(self):
        """Cleanup method called by atexit."""
        self.terminate_process(mode="graceful", delete_checkpoint=False)

    def get_training_model_config(self) -> dict | None:
        """Retrieves the model config for the current training run."""
        return (
            self.training_config.model_config if self.training_config else None
        )

    def get_training_data_config(self) -> dict | None:
        """Retrieves the data config for the current training run."""
        return (
            self.training_config.data_config if self.training_config else None
        )

    def update_config(self, training_config):
        self.training_config = training_config

    def write_model_config(self, model_config: ModelConfig):
        """Writes the model config to the model_config.json file."""
        with open(f"{self.work_dir}/model_config.json", "w") as f:
            json.dump(asdict(model_config), f)

    def start_training(self):
        """
        Starts the training process in a subprocess.
        """
        if not self.work_dir:
            st.error("Work directory is not set.")
            return

        if self.is_lock_file_locked():
            st.warning("Detected a lock file...")
            pid = self._read_lock_file()
            if pid and self._is_process_running(pid):
                st.warning("Training is already in progress.")
                return
            else:
                st.warning(
                    f"Process with PID {pid} is not running, removing lock file..."
                )
                self._remove_lock_file()

        if (
            not self.training_config
            or not self.training_config.data_config
            or not self.training_config.model_config
        ):
            st.error("Data or model configuration is not set.")
            return

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
            self.write_model_config(self.training_config.model_config)
            self._write_lock_file(self.training_process.pid)
            self._write_model_config(self.training_config.model_config)
        except Exception as e:
            st.error(f"Failed to start the training subprocess: {e}")
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
            shutil.rmtree(self.work_dir)
            self.reset_state()
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
        if delete_checkpoint:
            self.remove_work_dir()
        self.reset_state()
        return terminated

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

    def force_cleanup(self) -> bool:
        """Stop all managed processes and clean up files."""
        try:
            subprocess.run(
                ["pkill", "-f", config.TRAINER_MAIN_PATH], check=False
            )
            self.reset_state()
        except FileNotFoundError:
            print(
                "'pkill' command not found. Skipping orphaned process cleanup."
            )
            self.reset_state()
            return True
        except Exception as e:
            print(f"Error stopping processes: {e}")
            return False

    def _write_lock_file(self, pid: int) -> None:
        """Creates the training lock file."""
        try:
            with open(self.lock_file_path, "w") as f:
                f.write(str(pid))
        except OSError as e:
            print(
                f"Error: Could not create lock file at {self.lock_file_path}: {e}"
            )
            raise

    def _read_lock_file(self) -> int | None:
        """Reads the process ID from the lock file."""
        try:
            with open(self.lock_file_path, "r") as f:
                return int(f.read().strip())
        except (FileNotFoundError, ValueError):
            return None

    def _remove_lock_file(self) -> None:
        """Removes the training lock file if it exists."""
        if os.path.exists(self.lock_file_path):
            try:
                os.remove(self.lock_file_path)
            except OSError as e:
                print(
                    f"Error: Could not remove lock file at {self.lock_file_path}: {e}"
                )

    def is_lock_file_locked(self) -> bool:
        """Checks if the training lock file exists."""
        return os.path.exists(self.lock_file_path)

    def _open_log_files(self) -> tuple:
        """Opens training log files and returns the file handles."""
        try:
            self._log_stdout_handle = open(self.stdout_log_path, "w")
            self._log_stderr_handle = open(self.stderr_log_path, "w")
            return self._log_stdout_handle, self._log_stderr_handle
        except OSError as e:
            print(f"Error opening log files: {e}")
            if self._log_stdout_handle:
                self._log_stdout_handle.close()
            raise

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
        with open(f"{self.work_dir}/model_config.json", "w") as f:
            json.dump(asdict(model_config), f)

    def read_stdout_log(self) -> str:
        """Reads the content of the standard output log file."""
        try:
            with open(self.stdout_log_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def read_stderr_log(self) -> str:
        """Reads the content of the standard error log file."""
        try:
            with open(self.stderr_log_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    def get_status(self) -> TrainingStatus:
        """
        Investigates the lock file and OS to give a definitive status.
        This is the single source of truth for the system's state.
        """
        if self.training_process:
            if self.training_process.poll() is None:
                # Process is still running
                return TrainingStatus.RUNNING
            else:
                # Process has finished, reap the zombie and return FINISHED
                exit_code = self.training_process.wait()
                if exit_code == 0:
                    return TrainingStatus.FINISHED
                else:
                    return TrainingStatus.FAILED

        # If we don't have the process object (e.g., after a restart),
        # fall back to checking the lock file.
        if self.is_lock_file_locked():
            pid = self._read_lock_file()
            if pid and self._is_process_running(pid):
                return TrainingStatus.ORPHANED
            else:
                # Lock file exists but process is gone. It's a stale run.
                return TrainingStatus.FINISHED

        # No process object, no lock file. We are idle.
        return TrainingStatus.IDLE

    def reset_state(self, delete_checkpoint: bool = False) -> None:
        """
        Performs a full cleanup of files and resets the manager's
        internal state to idle.
        """
        self._close_log_files()
        self._remove_lock_file()
        if delete_checkpoint:
            self.remove_work_dir()
        self.training_process = None
        self.app_config = None
        self.work_dir = None

    def set_work_dir(self, work_dir: str | None) -> None:
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

    def remove_work_dir(self) -> None:
        """Removes the work directory."""
        if self.work_dir:
            shutil.rmtree(self.work_dir)
            self.work_dir = None
