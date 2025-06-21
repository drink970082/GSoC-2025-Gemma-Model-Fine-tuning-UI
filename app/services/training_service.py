import os
import time
from typing import Literal

from backend.manager.global_manager import (
    get_process_manager,
    get_status_manager,
    get_tensorboard_manager,
)
from config.app_config import get_config

config = get_config()


class TrainingService:
    """
    A service layer that abstracts the business logic for managing the
    training lifecycle. It acts as an intermediary between the UI and the
    backend managers.
    """

    def __init__(self):
        self.process_manager = get_process_manager()
        self.tensorboard_manager = get_tensorboard_manager()
        self.status_manager = get_status_manager()

    def start_training(self, app_config: dict) -> None:
        """
        Starts the training process and related services.
        """
        self.tensorboard_manager.reset_training_time()
        self.process_manager.update_config(
            app_config["data_config"], app_config["model_config"]
        )
        self.process_manager.start_training()

    def stop_training(
        self, mode: Literal["graceful", "force"] = "graceful"
    ) -> bool:
        """
        Stops all training-related processes.
        """
        self.tensorboard_manager.clear_cache()
        return self.process_manager.stop_all_processes(mode=mode)

    def wait_for_lock_file(self, timeout: int = 10) -> bool:
        """
        Waits for the training lock file to be created.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.process_manager.is_training_running():
                return True
            time.sleep(0.5)
        return False

    def is_training_running(self) -> bool:
        """Checks if a training process is currently active."""
        return self.process_manager.is_training_running()

    def get_training_status(self) -> str:
        """Gets the latest status message from the training process."""
        return self.status_manager.get()

    def get_model_config(self) -> dict | None:
        """Retrieves the model config for the current training run."""
        return self.process_manager.get_training_model_config()
