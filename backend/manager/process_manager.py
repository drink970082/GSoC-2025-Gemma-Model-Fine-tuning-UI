import json
import os
import signal
import subprocess
import time
from typing import Literal, Optional

import streamlit as st

from backend.manager.base_manager import BaseManager
from backend.manager.file_manager import FileManager
from config.app_config import get_config

config = get_config()


class ProcessManager(BaseManager):
    """A class to manage background subprocesses and cleanup operations."""

    def __init__(self):
        super().__init__()

    def initialize(self, file_manager: FileManager) -> None:
        """Initialize the ProcessManager."""
        self._initialized = True
        self.file_manager = file_manager
        self.training_process: Optional[subprocess.Popen] = None
        self.data_config = None
        self.model_config = None

    def cleanup(self):
        """Cleanup method called by atexit."""
        self.terminate_process(
            self.training_process, "training", mode="graceful"
        )

    def get_training_model_config(self) -> dict | None:
        """Retrieves the model config for the current training run."""
        return self.model_config

    def get_training_data_config(self) -> dict | None:
        """Retrieves the data config for the current training run."""
        return self.data_config

    def update_config(self, data_config, model_config):
        self.data_config = data_config
        self.model_config = model_config

    def clear_config(self):
        self.data_config = None
        self.model_config = None

    def start_training(self):
        """
        Starts the training process in a subprocess.
        """
        if self.file_manager.lock_file.is_locked():
            st.warning("Training is already in progress.")
            return

        if not self.data_config or not self.model_config:
            st.error("Data or model configuration is not set.")
            return

        data_config_str = str(self.data_config)
        model_config_str = str(self.model_config)

        command = [
            "python",
            config.TRAINER_MAIN_PATH,
            "--data_config",
            data_config_str,
            "--model_config",
            model_config_str,
        ]

        try:
            f_out, f_err = self.file_manager.log_files.open()
            self.training_process = subprocess.Popen(
                command, stdout=f_out, stderr=f_err
            )
            self.file_manager.lock_file.write(self.training_process.pid)
        except Exception as e:
            st.error(f"Failed to start the training subprocess: {e}")
            self.file_manager.log_files.close()
            return

        time.sleep(2)
        if self.training_process.poll() is not None:
            error_output = self.file_manager.log_files.read_stderr()
            if error_output:
                st.error("The training process failed to start. Error:")
                st.code(error_output, language="bash")
            else:
                st.error(
                    "The training process failed to start with no error message."
                )

            # Clean up lock file and close logs if process dies immediately
            self.file_manager.lock_file.remove()
            self.file_manager.log_files.close()
            return

    def terminate_process(
        self,
        process: Optional[subprocess.Popen],
        name: str,
        mode: Literal["graceful", "force"] = "graceful",
    ) -> bool:
        """Terminate a process either gracefully or forcefully.

        Args:
            process: The process to terminate
            name: Name of the process for logging
            mode: Either "graceful" (SIGINT) or "force" (SIGKILL)

        Returns:
            bool: True if process was terminated successfully
        """
        terminated = True
        if not process or process.poll() is not None:
            return True

        if mode == "graceful":
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return self._terminate_process(process, name, mode="force")
        else:
            try:
                process.kill()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                terminated = False
        self.file_manager.cleanup()
        return terminated

    def force_cleanup(self) -> bool:
        """Stop all managed processes and clean up files.

        Args:
            mode: Either "graceful" (SIGINT) or "force" (SIGKILL)

        Returns:
            bool: True if all processes were stopped successfully
        """
        try:
            subprocess.run(
                ["pkill", "-f", config.TRAINER_MAIN_PATH], check=False
            )
        except FileNotFoundError:
            print(
                "'pkill' command not found. Skipping orphaned process cleanup."
            )

            self.file_manager.cleanup()
            return True
        except Exception as e:
            print(f"Error stopping processes: {e}")
            return False
