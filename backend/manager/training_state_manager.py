import json
import os
import tempfile
from enum import Enum, auto
from typing import Any, Dict, Optional

from backend.manager.base_manager import BaseManager
from config.app_config import get_config

config = get_config()





class TrainingStateManager(BaseManager):
    def __init__(self):
        super().__init__()
        self.state_file = config.TRAINING_STATE_FILE


    def _atomic_write(self, data: dict):
        # Write to temp file, then rename for atomicity
        dir_name = os.path.dirname(self.state_file)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tf:
            json.dump(data, tf, indent=2)
            tempname = tf.name
        os.replace(tempname, self.state_file)

    def get_state(self) -> Dict[str, Any]:
        if not self.state_file or not os.path.exists(self.state_file):
            return {"status": "IDLE"}
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, IOError):
            return {"status": "IDLE"}

    def set_state(
        self,
        status: str,
        pid: Optional[int] = None,
        error: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ):
        state = {
            "status": status,
            "pid": pid,
            "error": error,
            "start_time": start_time,
            "end_time": end_time,
        }
        self._atomic_write(state)

    def mark_running(self, pid: int, start_time: str):
        self.set_state(
            "RUNNING",
            pid=pid,
            start_time=start_time,
        )

    def mark_finished(self, end_time: str):
        state = self.get_state()
        self.set_state(
            "FINISHED",
            end_time=end_time,
            **{k: state.get(k) for k in ["pid", "start_time"]}
        )

    def mark_failed(self, error: str, end_time: str):
        state = self.get_state()
        self.set_state(
            "FAILED",
            error=error,
            end_time=end_time,
            **{k: state.get(k) for k in ["pid", "start_time"]}
        )

    def mark_idle(self):
        self.set_state("IDLE")

    def mark_orphaned(self, error: str):
        state = self.get_state()
        self.set_state(
            "ORPHANED",
            error=error,
            **{k: state.get(k) for k in ["pid", "start_time"]}
        )

    def cleanup(self):
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
