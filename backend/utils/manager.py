import os
import signal
import subprocess
import time
from typing import Literal, Optional

import streamlit as st

from config.training_config import (
    LOCK_FILE,
    STATUS_LOG,
    TENSORBOARD_LOGDIR,
    TENSORBOARD_PORT,
    TRAINER_MAIN_PATH,
    TRAINER_STDERR_LOG,
    TRAINER_STDOUT_LOG,
)


class StatusManager:
    """Manages the training status updates and file handling."""

    def __init__(self, status_file: str = STATUS_LOG):
        self.status_file = status_file

    def update(self, message: str) -> None:
        """Update the status message in the status file."""
        with open(self.status_file, "w") as f:
            f.write(message)

    def cleanup(self) -> None:
        """Clean up the status file if it exists."""
        if os.path.exists(self.status_file):
            os.remove(self.status_file)


class ProcessManager:
    """A class to manage background subprocesses and cleanup operations."""

    def __init__(self):
        self.training_process: Optional[subprocess.Popen] = None
        self.tensorboard_process: Optional[subprocess.Popen] = None
        print("ProcessManager initialized.")

    def start_training(self, data_config, model_config):
        if os.path.exists(LOCK_FILE):
            st.warning("Training is already in progress.")
            return
        st.info("Fine-tuning has started. This may take a while.")
        command = [
            "python",
            TRAINER_MAIN_PATH,
            "--data_config",
            str(data_config),
            "--model_config",
            str(model_config),
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

        print(f"Terminating {name} process (PID: {process.pid})...")

        if mode == "graceful":
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
                print(f"{name} process terminated gracefully.")
                return True
            except subprocess.TimeoutExpired:
                print(f"{name} did not terminate gracefully. Forcing kill.")
                return self._terminate_process(process, name, mode="force")
        else:
            try:
                process.kill()
                process.wait(timeout=2)
                print(f"{name} process forcefully terminated.")
                return True
            except subprocess.TimeoutExpired:
                print(f"Failed to forcefully terminate {name} process.")
                return False

    def _clean_files(self):
        """Clean up all temporary files."""
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
                    print(f"Removed file: {f}")
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
                        ["pkill", "-f", TRAINER_MAIN_PATH], check=False
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
