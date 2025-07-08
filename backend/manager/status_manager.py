from backend.manager.base_manager import BaseManager
import os
from config.app_config import get_config

config = get_config()


class StatusManager(BaseManager):
    """Manages the training status updates and file handling."""

    def __init__(self):
        super().__init__()
        self.status_file_path = config.STATUS_LOG

    def cleanup(self):
        """Removes the status log file."""
        pass

    def update(self, message: str) -> None:
        """Update the status message in the status file."""
        if not self.status_file_path:
            return
        try:
            with open(self.status_file_path, "w") as f:
                f.write(message)
        except OSError as e:
            print(f"Error writing to status file: {e}")

    def get(self) -> str:
        """Reads the current status from the status file."""
        if not self.status_file_path:
            return "Idle"
        try:
            with open(self.status_file_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "Idle"

    def set_work_dir(self, work_dir: str | None) -> None:
        """Set the work directory and derive file paths."""
        super().set_work_dir(work_dir)
        if work_dir:
            self.status_file_path = os.path.join(work_dir, config.STATUS_LOG)
        else:
            self.status_file_path = None
