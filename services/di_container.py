import atexit
from typing import Any, Dict, List

from backend.manager.process_manager import ProcessManager
from backend.manager.training_state_manager import TrainingStateManager
from backend.manager.system_manager import SystemManager
from backend.manager.tensorboard_manager import TensorBoardManager


class DIContainer:
    """Simple dependency injection container with built-in setup and cleanup."""

    def __init__(self) -> None:
        """Initialize the container with empty services registry."""
        self._services: Dict[str, Any] = {}
        self._setup_done = False

    def get(self, name: str) -> Any:
        """Get a service by name, setting up services if needed."""
        if not self._setup_done:
            self._setup_services()

        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")

        return self._services[name]

    def _register(self, name: str, service: Any) -> None:
        """Register a service with the given name."""
        self._services[name] = service

    def _setup_services(self) -> None:
        """Set up all services in the container."""
        if self._setup_done:
            return

        training_state_manager = TrainingStateManager()
        process_manager = ProcessManager(training_state_manager)
        tensorboard_manager = TensorBoardManager()
        system_manager = SystemManager()

        self._register("process_manager", process_manager)
        self._register("training_state_manager", training_state_manager)
        self._register("tensorboard_manager", tensorboard_manager)
        self._register("system_manager", system_manager)

        from .training_service import TrainingService

        training_service = TrainingService(
            process_manager=process_manager,
            tensorboard_manager=tensorboard_manager,
            system_manager=system_manager,
        )
        self._register("training_service", training_service)
        atexit.register(self._cleanup_all)
        self._setup_done = True

    def _cleanup_all(self) -> None:
        """Cleanup all services that have cleanup methods."""
        print("ATEIXT: Cleaning up all services...")
        for service in self._services.values():
            if hasattr(service, "cleanup"):
                try:
                    service.cleanup()
                except Exception as e:
                    print(f"Error during cleanup: {e}")


# Module-level container instance
_container = DIContainer()


def get_service(name: str) -> Any:
    """Convenience function to get a service from the global container."""
    return _container.get(name)
