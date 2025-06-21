import json
import os
import signal
import subprocess
import time
from typing import Literal, Optional

import streamlit as st

from config.app_config import get_config

config = get_config()


class ProcessManager:
    """A class to manage background subprocesses and cleanup operations."""

    def __init__(self):
        self.training_process: Optional[subprocess.Popen] = None
        self.tensorboard_process: Optional[subprocess.Popen] = None
        self.data_config = None
        self.model_config = None

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
        if os.path.exists(config.LOCK_FILE):
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
            with open(config.TRAINER_STDOUT_LOG, "w") as f_out, open(
                config.TRAINER_STDERR_LOG, "w"
            ) as f_err:
                self.training_process = subprocess.Popen(
                    command, stdout=f_out, stderr=f_err
                )
        except Exception as e:
            st.error(f"Failed to start the training subprocess: {e}")
            return
        time.sleep(2)
        if self.training_process.poll() is not None:
            try:
                with open(config.TRAINER_STDERR_LOG, "r") as f:
                    error_output = f.read()
                if error_output:
                    st.error("The training process failed to start. Error:")
                    st.code(error_output, language="bash")
                else:
                    st.error(
                        "The training process failed to start with no error message."
                    )
            except FileNotFoundError:
                st.error(
                    "Could not read the error log file for the training process."
                )

            if os.path.exists(config.LOCK_FILE):
                os.remove(config.LOCK_FILE)
            return

    def start_tensorboard(self):
        if self.tensorboard_process and self.tensorboard_process.poll() is None:
            return
        command = [
            "tensorboard",
            "--logdir",
            config.TENSORBOARD_LOGDIR,
            "--port",
            str(config.TENSORBOARD_PORT),
        ]
        self.tensorboard_process = subprocess.Popen(command)
        time.sleep(5)

    def _terminate_process(
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
        if not process or process.poll() is not None:
            return True

        if mode == "graceful":
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
                return True
            except subprocess.TimeoutExpired:
                return self._terminate_process(process, name, mode="force")
        else:
            try:
                process.kill()
                process.wait(timeout=2)
                return True
            except subprocess.TimeoutExpired:
                return False

    def _clean_files(self):
        """Clean up all temporary files."""
        files_to_delete = [
            config.LOCK_FILE,
            config.STATUS_LOG,
            config.TRAINER_STDOUT_LOG,
            config.TRAINER_STDERR_LOG,
        ]

        for f in files_to_delete:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except OSError as e:
                print(f"Could not remove file {f}: {e}")

    def stop_all_processes(
        self, mode: Literal["graceful", "force"] = "graceful"
    ) -> bool:
        """Stop all managed processes and clean up files.

        Args:
            mode: Either "graceful" (SIGINT) or "force" (SIGKILL)

        Returns:
            bool: True if all processes were stopped successfully
        """
        try:
            training_stopped = self._terminate_process(
                self.training_process, "Training", mode
            )
            tensorboard_stopped = self._terminate_process(
                self.tensorboard_process, "TensorBoard", mode
            )

            if mode == "force":
                # If in force mode, also try to kill any orphaned processes
                try:
                    subprocess.run(
                        ["pkill", "-f", config.TRAINER_MAIN_PATH], check=False
                    )
                    subprocess.run(["pkill", "-f", "tensorboard"], check=False)
                except FileNotFoundError:
                    print(
                        "'pkill' command not found. Skipping orphaned process cleanup."
                    )

            self._clean_files()
            return training_stopped and tensorboard_stopped
        except Exception as e:
            print(f"Error stopping processes: {e}")
            return False

    def cleanup(self):
        """Cleanup method called by atexit."""
        print("ATEIXT: Shutting down background processes...")
        self.stop_all_processes(mode="graceful")

    @staticmethod
    def is_training_running() -> bool:
        """Check if training is currently in progress by checking the lock file."""
        return os.path.exists(config.LOCK_FILE)
