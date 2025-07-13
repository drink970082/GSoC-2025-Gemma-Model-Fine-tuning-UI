import json
import os
import tempfile
from typing import Any, Optional, Dict

from backend.manager.base_manager import BaseManager
from config.app_config import get_config, TrainingStatus

config = get_config()


class TrainingStateManager(BaseManager):
    """Manages the training state."""

    def __init__(self) -> None:
        """Initialize the TrainingStateManager."""
        super().__init__()
        self.state_file: str = config.TRAINING_STATE_FILE

    def _atomic_write(self, data: dict[str, Any]) -> None:
        """Write to temp file, then rename for atomicity."""
        dir_name = os.path.dirname(self.state_file)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
            json.dump(data, tf, indent=2)
            tempname = tf.name
        os.replace(tempname, self.state_file)

    def get_state(self) -> Dict[str, Any]:
        """Get the training state."""
        if not self.state_file or not os.path.exists(self.state_file):
            return {"status": TrainingStatus.IDLE.value}
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"status": TrainingStatus.IDLE.value}

    def set_state(
        self,
        status: TrainingStatus,
        pid: Optional[int] = None,
        error: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> None:
        """Set the training state."""
        state = {
            "status": status.value,
            "pid": pid,
            "error": error,
            "start_time": start_time,
            "end_time": end_time,
        }
        self._atomic_write(state)

    def mark_running(self, pid: int, start_time: str) -> None:
        """Mark the training as running."""
        self.set_state(
            TrainingStatus.RUNNING,
            pid=pid,
            start_time=start_time,
        )

    def mark_finished(self, end_time: str) -> None:
        """Mark the training as finished."""
        state = self.get_state()
        self.set_state(
            TrainingStatus.FINISHED,
            end_time=end_time,
            **{k: state.get(k) for k in ["pid", "start_time"]}
        )

    def mark_failed(self, error: str, end_time: str) -> None:
        """Mark the training as failed."""
        state = self.get_state()
        self.set_state(
            TrainingStatus.FAILED,
            error=error,
            end_time=end_time,
            **{k: state.get(k) for k in ["pid", "start_time"]}
        )

    def mark_idle(self) -> None:
        """Mark the training as idle."""
        self.set_state(TrainingStatus.IDLE)

    def mark_orphaned(self, error: str) -> None:
        """Mark the training as orphaned."""
        state = self.get_state()
        self.set_state(
            TrainingStatus.ORPHANED,
            error=error,
            **{k: state.get(k) for k in ["pid", "start_time"]}
        )

    def cleanup(self) -> None:
        """Clean up the training state."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
