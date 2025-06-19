import os

from config.training_config import STATUS_LOG


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
