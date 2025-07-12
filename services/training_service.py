import os
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import streamlit as st

from backend.manager.process_manager import ProcessManager
from backend.manager.system_manager import SystemManager
from backend.manager.tensorboard_manager import TensorBoardManager
from backend.manager.training_state_manager import TrainingStateManager
from config.app_config import get_config
from config.dataclass import TrainingConfig

config = get_config()


class TrainingService:
    """
    Service layer for training operations, using the lock file as the single source of truth.
    """

    def __init__(
        self,
        process_manager: ProcessManager,
        tensorboard_manager: TensorBoardManager,
        system_manager: SystemManager,
        training_state_manager: TrainingStateManager,
    ):
        """Initialize the training service with DI container."""
        self.process_manager = process_manager
        self.tensorboard_manager = tensorboard_manager
        self.system_manager = system_manager
        self.training_state_manager = training_state_manager
        self.training_config = None
        self.work_dir = None

    def start_training(self, training_config: TrainingConfig) -> None:
        """
        Starts the training process, using the lock file to prevent duplicates.
        """
        state = self.training_state_manager.get_state()
        if state.get("status") == "RUNNING":
            st.warning("Training is already in progress.")
            return
        
        self.training_config = training_config
        self.process_manager.update_config(training_config)
        self._set_work_dir(training_config.model_name)
        self.tensorboard_manager.reset_training_time()
        self.process_manager.start_training()

    def wait_for_state_file(self, timeout: int = 10) -> bool:
        """Wait for the lock file to be created."""
        for _ in range(timeout):
            state = self.training_state_manager.get_state()
            if state.get("status") == "RUNNING":
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
                mode=mode, delete_checkpoint=True
            )
            self._reset_work_dir()
            self.tensorboard_manager.reset_training_time()
            return processes_stopped
        except Exception as e:
            st.error(f"Failed to stop training: {e}")
            return False
        
        

    def is_training_running(self) -> str:
        """
        Checks training status using the state manager as the single source of truth.
        Handles orphan detection and cleanup.
        """
        state = self.training_state_manager.get_state()
        status = state.get("status", "IDLE")

        # Handle orphaned processes
        if status == "ORPHANED":
            st.warning(
                "An orphaned training process from a previous session was detected. "
                "Automatically cleaning up...."
            )
            self.process_manager.force_cleanup()
            st.success("Cleanup complete. The application is now ready.")
            time.sleep(2)
            return "IDLE"

        # Handle finished training
        if status == "FINISHED":
            self.process_manager.reset_state()
            return "FINISHED"

        # Handle failed training
        if status == "FAILED":
            self.process_manager.reset_state(delete_checkpoint=True)
            return "FAILED"

        return status

    def get_training_config(self) -> Optional[TrainingConfig]:
        """Retrieve the training configuration for the current training run."""
        return self.training_config

    def get_kpi_data(self) -> dict:
        """
        Gathers all relevant latest metrics and metadata for the UI KPI panel.
        """
        # Ensure the latest data is loaded from the event file
        self.tensorboard_manager._get_data()
        total_steps = self.training_config.model_config.epochs

        return {
            # Metadata
            **self.tensorboard_manager.get_parsed_metadata().get(
                "parameters", {}
            ),
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
        stdout = self.process_manager.read_stdout_log()
        stderr = self.process_manager.read_stderr_log()
        return stdout, stderr

    def _set_work_dir(self, model_name: str) -> None:
        """Updates the work directory with the model configuration."""
        self.work_dir = f"{model_name}-{time.strftime('%Y%m%d_%H%M%S')}"
        self.work_dir = os.path.join(config.CHECKPOINT_FOLDER, self.work_dir)
        self.process_manager.set_work_dir(self.work_dir)
        self.tensorboard_manager.set_work_dir(self.work_dir)
    def _reset_work_dir(self) -> None:
        """Resets the work directory."""
        self.work_dir = None
        self.process_manager.set_work_dir(self.work_dir)
        self.tensorboard_manager.set_work_dir(self.work_dir)