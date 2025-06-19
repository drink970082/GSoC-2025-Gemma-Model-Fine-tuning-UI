import atexit
from ProcessManager import ProcessManager
from TensorboardManager import TensorBoardDataManager

# Global managers
_process_manager = None
_tensorboard_manager = None


def get_process_manager():
    """Get or create global process manager."""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
        atexit.register(_process_manager.cleanup)
    return _process_manager


def get_tensorboard_manager():
    """Get or create global tensorboard manager."""
    global _tensorboard_manager
    if _tensorboard_manager is None:
        _tensorboard_manager = TensorBoardDataManager()
    return _tensorboard_manager


def cleanup_managers():
    """Cleanup all global managers."""
    global _process_manager, _tensorboard_manager
    if _process_manager:
        _process_manager.cleanup()
        _process_manager = None
    if _tensorboard_manager:
        _tensorboard_manager.clear_cache()
        _tensorboard_manager = None
