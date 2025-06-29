"""
Training Service with DI Container Integration

This service uses the DI container to resolve its dependencies,
making it simpler while maintaining the same functionality.
"""

import time
from typing import Any, Dict, Literal, Optional

import streamlit as st

from services.di_container import get_container


class TrainingService:
    """
    Service layer for training operations, using the lock file as the single source of truth.
    """

    def __init__(self, container):
        """Initialize the training service with DI container."""
        self.container = container

    @property
    def process_manager(self):
        """Get the process manager from the container."""
        return self.container.get("process_manager")

    @property
    def file_manager(self):
        """Get the file manager from the container."""
        return self.container.get("file_manager")

    @property
    def tensorboard_manager(self):
        """Get the tensorboard manager from the container."""
        return self.container.get("tensorboard_manager")

    @property
    def status_manager(self):
        """Get the status manager from the container."""
        return self.container.get("status_manager")

    def start_training(self, app_config: Dict[str, Any]) -> None:
        """
        Starts the training process, using the lock file to prevent duplicates.
        """
        if self.is_training_running():
            st.warning("Training is already in progress.")
            return

        self.process_manager.update_config(
            app_config["data_config"], app_config["model_config"]
        )
        self.tensorboard_manager.reset_training_time()
        self.file_manager.create_lock_file()
        self.process_manager.start_training()

    def stop_training(
        self, mode: Literal["graceful", "force"] = "graceful"
    ) -> bool:
        """
        Stops the training process and always cleans up the lock file.
        """
        try:
            processes_stopped = self.process_manager.stop_all_processes(
                mode=mode
            )
            self.tensorboard_manager.clear_cache()
            return processes_stopped
        finally:
            # This is critical: always remove the lock file and clean up.
            self.file_manager.clean_training_files()

    def is_training_running(self) -> bool:
        """
        Checks if a training run is active by only checking for the lock file.
        This is the single source of truth for the application's training state.
        """
        return self.file_manager.is_training_locked()

    def get_training_status(self) -> str:
        """Gets the latest status message from the training process."""
        return self.status_manager.get()

    def get_model_config(self) -> Optional[Dict[str, Any]]:
        """Retrieve the model configuration for the current training run."""
        return self.process_manager.get_training_model_config()

    def get_data_config(self) -> Optional[Dict[str, Any]]:
        """Retrieve the data configuration for the current training run."""
        return self.process_manager.get_training_data_config()

    def get_tensorboard_data(self) -> Dict[str, Any]:
        """Get the latest TensorBoard data and metrics."""
        return self.tensorboard_manager.get_data()

    def get_training_metrics(self) -> Dict[str, Any]:
        """Get current training metrics and performance data."""
        return self.tensorboard_manager.get_training_metrics()

    def get_latest_values(self) -> Dict[str, Any]:
        """Get the latest values for all training metrics."""
        return self.tensorboard_manager.get_latest_values()

    def clear_tensorboard_cache(self) -> None:
        """Clear the TensorBoard data cache."""
        self.tensorboard_manager.clear_cache()

    def reset_training_time(self) -> None:
        """Reset the training start time tracking."""
        self.tensorboard_manager.reset_training_time()

    def force_cleanup(self):
        """
        Forcibly stops all processes and cleans all related files.
        This is an escape hatch for a stuck state, e.g., a stale lock file.
        """
        print("Forcing cleanup of all training resources...")
        try:
            self.process_manager.stop_all_processes(mode="force")
        finally:
            # Always ensure files are cleaned even if process stopping fails
            self.file_manager.clean_training_files()
        print("Forced cleanup complete.")
