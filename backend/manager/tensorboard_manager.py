import time
import pandas as pd
from typing import Any, Optional, Dict

from backend.manager.base_manager import BaseManager
from backend.utils.tensorboard_event_parser import EventFileParser
from config.app_config import get_config

config = get_config()


class TensorBoardManager(BaseManager):
    """Manages TensorBoard event data loading and caching."""

    def __init__(self) -> None:
        """Initialize the TensorBoardManager."""
        super().__init__()
        self._event_data: Dict[str, Any] = {}
        self.tensorboard_log_dir: Optional[str] = None
        self.event_file_parser = EventFileParser(self.tensorboard_log_dir)

    def cleanup(self) -> None:
        """Cleanup method called by atexit."""
        self._event_data = {}

    def get_kpi_data(self, total_steps: int) -> Dict[str, Any]:
        """Get KPI data."""
        try:
            data = self.event_file_parser.load_event_data()
        except Exception as e:
            raise ValueError(f"Error loading event data: {e}")
        if not data:
            return {}
        metadata = data['metadata']
        training_metrics = data['training_metrics']
        latest_training_metrics = data['latest_training_metrics']
        training_speed = latest_training_metrics.get(
            "perf_stats/steps_per_sec", 0.0
        )
        current_step = (
            training_metrics["losses/loss"].iloc[-1]["step"]
            if not training_metrics.get("losses/loss", pd.DataFrame()).empty
            else 0
        )
        eta_str = self._get_eta_str(total_steps, training_speed, current_step)
        return {
            # Metadata
            "total_params": metadata.get("parameters", {}).get("total_params", 0),
            "total_bytes": metadata.get("parameters", {}).get("total_bytes", 0),
            "total_layers": len(metadata.get("parameters", {}).get("layers", [])),
            # Progress KPIs
            "current_step": current_step,
            "total_steps": total_steps,
            "current_loss": latest_training_metrics.get("losses/loss", 0.0),
            "training_time": latest_training_metrics.get(
                "perf_stats/total_training_time_hours", 0.0
            ),
            # Performance KPIs
            "training_speed": training_speed,
            "data_throughput": latest_training_metrics.get(
                "perf_stats/data_points_per_sec_global", 0.0
            ),
            "avg_step_time": latest_training_metrics.get(
                "perf_stats/train/avg_time_sec", 0.0
            ),
            "avg_eval_time": latest_training_metrics.get(
                "perf_stats/evals_along_train/avg_time_sec", 0.0
            ),
            "eta_str": eta_str,
        }

    def get_training_metrics(self) -> Dict[str, pd.DataFrame]:
        """Get the training metrics from the TensorBoard manager."""
        try:
            data = self.event_file_parser.load_event_data()
        except Exception as e:
            raise ValueError(f"Error loading event data: {e}")
        return data.get('training_metrics', {})

    def _get_eta_str(
        self, total_steps: int, training_speed: float, current_step: int
    ) -> str:
        """Get the estimated time remaining as a formatted string."""
        if training_speed > 0 and total_steps > 0:
            remaining_steps = total_steps - current_step
            if remaining_steps > 0:
                eta_seconds = remaining_steps / training_speed
                return time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
        return "N/A"

    def set_work_dir(self, work_dir: str | None) -> None:
        """Set the work directory for TensorBoard logs."""
        super().set_work_dir(work_dir)
        self.tensorboard_log_dir = work_dir
        self.event_file_parser.set_work_dir(work_dir)
