from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseManager(ABC):
    """Base class for all managers with standard lifecycle."""

    def __init__(self, config: Any = None):
        self.config = config

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
