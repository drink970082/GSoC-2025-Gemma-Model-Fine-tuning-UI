import atexit
from typing import Any, Dict

from backend.manager.process_manager import ProcessManager
from backend.manager.training_state_manager import TrainingStateManager
from backend.manager.system_manager import SystemManager
from backend.manager.tensorboard_manager import TensorBoardManager


class DIContainer:
    """
    Simple dependency injection container with built-in setup and cleanup.
    """

    def __init__(self) -> None:
        """Initialize the container with empty services registry."""
        self._services: Dict[str, Any] = {}
        self._setup_done = False

    def register(self, name: str, service: Any) -> None:
        """
        Register a service with the given name.

        Args:
            name: Service identifier
            service: Service instance to register
        """
        self._services[name] = service

    def get(self, name: str) -> Any:
        """
        Get a service by name, setting up services if needed.

        Args:
            name: Service identifier

        Returns:
            The registered service instance

        Raises:
            KeyError: If service is not registered
        """
        # Auto-setup on first access
        if not self._setup_done:
            self._setup_services()

        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")

        return self._services[name]

    def is_registered(self, name: str) -> bool:
        """
        Check if a service is registered.

        Args:
            name: Service identifier

        Returns:
            True if service is registered, False otherwise
        """
        return name in self._services

    def list_services(self) -> list:
        """
        Get list of all registered service names.

        Returns:
            List of service names
        """
        return list(self._services.keys())

    def _setup_services(self) -> None:
        """
        Set up all services in the container.

        This is called automatically on first service access,
        ensuring proper initialization order.
        """
        if self._setup_done:
            return

        # Create managers
        training_state_manager = TrainingStateManager()
        process_manager = ProcessManager(training_state_manager)
        tensorboard_manager = TensorBoardManager()
        system_manager = SystemManager()

        self.register("process_manager", process_manager)
        self.register("training_state_manager", training_state_manager)
        self.register("tensorboard_manager", tensorboard_manager)
        self.register("system_manager", system_manager)

        # Create and register training service with container dependency
        from .training_service import TrainingService

        training_service = TrainingService(
            process_manager=process_manager,
            tensorboard_manager=tensorboard_manager,
            training_state_manager=training_state_manager,
            system_manager=system_manager,
        )
        self.register("training_service", training_service)

        # Register cleanup handler once during setup
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


def get_container() -> DIContainer:
    """
    Get the global container instance.

    This preserves the singleton behavior you need for Streamlit,
    ensuring managers persist across st.rerun() calls.

    Returns:
        The global DIContainer instance
    """
    return _container


def get_service(name: str) -> Any:
    """
    Convenience function to get a service from the global container.

    Args:
        name: Service identifier

    Returns:
        The resolved service instance
    """
    return get_container().get(name)
