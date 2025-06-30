import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import streamlit as st
from backend.manager.process_manager import ProcessManager
from backend.manager.tensorboard_manager import TensorBoardManager
from backend.manager.status_manager import StatusManager
from backend.manager.file_manager import FileManager
from backend.manager.system_manager import SystemManager


class TrainingService:
    """
    Service layer for training operations, using the lock file as the single source of truth.
    """

    def __init__(
        self,
        process_manager: ProcessManager,
        tensorboard_manager: TensorBoardManager,
        status_manager: StatusManager,
        file_manager: FileManager,
        system_manager: SystemManager,
    ):
        """Initialize the training service with DI container."""
        self.process_manager = process_manager
        self.tensorboard_manager = tensorboard_manager
        self.status_manager = status_manager
        self.file_manager = file_manager
        self.system_manager = system_manager

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
        self.process_manager.start_training()

    def wait_for_lock_file(self, timeout: int = 10) -> bool:
        """Wait for the lock file to be created."""
        for _ in range(timeout):
            if self.file_manager.lock_file.is_locked():
                return True
            time.sleep(1)
        st.error("Training process did not start within timeout.")
        return False

    def stop_training(
        self, mode: Literal["graceful", "force"] = "graceful"
    ) -> bool:
        """
        Stops the training process and always cleans up the lock file.
        """
        try:
            processes_stopped = self.process_manager.terminate_process(
                mode=mode
            )
            self.tensorboard_manager.reset_training_time()
            return processes_stopped
        except Exception as e:
            st.error(f"Failed to stop training: {e}")
            return False

    def is_training_running(self) -> bool:
        """
        Checks if a training run is active by only checking for the lock file.
        This is the single source of truth for the application's training state.
        """
        return self.file_manager.lock_file.is_locked()

    def get_training_status(self) -> str:
        """Gets the latest status message from the training process."""
        return self.status_manager.get()

    def get_model_config(self) -> Optional[Dict[str, Any]]:
        """Retrieve the model configuration for the current training run."""
        return self.process_manager.get_training_model_config()

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the latest checkpoint."""
        return self.file_manager.checkpoint_file.find_latest_checkpoint()

    def get_kpi_data(self) -> dict:
        """
        Gathers all relevant latest metrics and metadata for the UI KPI panel.
        """
        # Ensure the latest data is loaded from the event file
        self.tensorboard_manager._get_data()

        model_config = self.get_model_config() or {}
        total_steps = model_config.get("epochs", 0)

        return {
            # Metadata
            **self.tensorboard_manager.get_parsed_metadata(),
            # Progress KPIs
            "current_step": self.tensorboard_manager.get_current_step(),
            "total_steps": total_steps,
            "current_loss": self.tensorboard_manager.get_current_loss(),
            "training_time": self.tensorboard_manager.get_training_time(),
            # Performance KPIs
            "training_speed": self.tensorboard_manager.get_training_speed(),
            "data_throughput": self.tensorboard_manager.get_data_throughput(),
            "avg_step_time": self.tensorboard_manager.get_avg_step_time(),
            "avg_eval_time": self.tensorboard_manager.get_avg_eval_time(),
            "eta_str": self.tensorboard_manager.get_eta_str(total_steps),
        }

    def get_tensorboard_data(self) -> Dict[str, Any]:
        """Get the latest TensorBoard data and metrics."""
        return self.tensorboard_manager._get_data()

    def get_loss_metrics(self) -> dict:
        """Get the loss metrics from the TensorBoard manager."""
        return self.tensorboard_manager.get_loss_metrics()

    def get_performance_metrics(self) -> dict:
        """Get the performance metrics from the TensorBoard manager."""
        return self.tensorboard_manager.get_performance_metrics()

    def poll_system_usage(self) -> None:
        """A pass-through method to poll system usage via the SystemManager."""
        self.system_manager.poll_system_usage()

    def get_system_usage_history(self) -> dict:
        """A pass-through method to get system usage history from the SystemManager."""
        return self.system_manager.get_history_as_dataframes()

    def has_gpu(self) -> bool:
        """A pass-through method to check if a GPU is available for monitoring."""
        return self.system_manager.has_gpu()

    def get_log_contents(self) -> tuple[str, str]:
        """Returns the contents of the stdout and stderr log files."""
        stdout = self.file_manager.log_files.read_stdout()
        stderr = self.file_manager.log_files.read_stderr()
        return stdout, stderr
