from backend.manager.base_manager import BaseManager
from backend.manager.file_manager import FileManager


class StatusManager(BaseManager):
    """Manages the training status updates and file handling."""

    def __init__(self):
        super().__init__()

    def initialize(self, file_manager: FileManager):
        """Initialize the StatusManager."""
        self._initialized = True
        self.file_manager = file_manager

    def update(self, message: str) -> None:
        """Update the status message in the status file."""
        self.file_manager.status_file.write(message)

    def get(self) -> str:
        """Get the current training status from the status file."""
        return self.file_manager.status_file.read()
