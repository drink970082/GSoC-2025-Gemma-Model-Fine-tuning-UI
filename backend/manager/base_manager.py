from abc import ABC, abstractmethod
from typing import Any, Dict
import os


class BaseManager(ABC):
    """Base class for all managers with standard lifecycle."""

    def __init__(self, config: Any = None):
        self.config = config
        self.work_dir = None

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass

    def health_check(self) -> Dict[str, Any]:
        """Return basic health info."""
        return {
            "initialized": self._initialized,
            "manager": self.__class__.__name__,
        }

    def set_work_dir(self, work_dir: str) -> None:
        """set the work directory."""
        # We must create the work_dir if it doesn't exist
        os.makedirs(work_dir, exist_ok=True)
        self.work_dir = work_dir
