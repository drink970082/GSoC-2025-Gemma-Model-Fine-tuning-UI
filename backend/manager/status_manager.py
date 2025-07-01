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
        if os.path.exists(self.status_file_path):
            os.remove(self.status_file_path)

    def update(self, message: str) -> None:
        """Update the status message in the status file."""
        try:
            with open(self.status_file_path, "w") as f:
                f.write(message)
        except OSError as e:
            print(f"Error writing to status file: {e}")

    def get(self) -> str:
        """Reads the current status from the status file."""
        try:
            with open(self.status_file_path, "r") as f:
                return f.read().strip()
        except FileNotFoundError:
            return "Initializing"
