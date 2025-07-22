from abc import ABC, abstractmethod
from typing import Any, Optional
import os


class BaseManager(ABC):
    """Base class for all managers with standard lifecycle."""

    def __init__(self, config: Any = None) -> None:
        """Initialize the BaseManager."""
        self.config: Any = config
        self.work_dir: Optional[str] = None

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def set_work_dir(self, work_dir: Optional[str]) -> None:
        """set the work directory."""
        # We must create the work_dir if it doesn't exist
        if work_dir:
            os.makedirs(work_dir, exist_ok=True)
            self.work_dir = work_dir
        else:
            self.work_dir = None
