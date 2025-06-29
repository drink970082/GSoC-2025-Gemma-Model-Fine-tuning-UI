import os

from config.app_config import get_config

from .base_manager import BaseManager

config = get_config()


class FileManager(BaseManager):
    """
    Manages file system operations via specialized sub-managers.
    Acts as a container for LockFileManager, StatusFileManager, and LogFileManager.
    """

    def __init__(self):
        super().__init__()
        self.lock_file = None
        self.status_file = None
        self.log_files = None
        self.tensorboard_file = None

    def initialize(self) -> None:
        """Initialize the FileManager and its sub-managers."""
        self._initialized = True
        self.lock_file = self.LockFileManager()
        self.status_file = self.StatusFileManager()
        self.log_files = self.LogFileManager()
        self.tensorboard_file = self.TensorBoardFileManager()

    def cleanup(self) -> None:
        """Clean up all managed files upon application exit."""
        self.lock_file.remove()
        self.status_file.remove()
        self.log_files.remove()
        self.tensorboard_file.remove()

    # --- Helper Classes ---

    class LockFileManager:
        """Manages the training lock file."""

        def write(self, pid: int) -> None:
            """Creates the training lock file."""
            try:
                with open(config.LOCK_FILE, "w") as f:
                    f.write(str(pid))
            except OSError as e:
                print(
                    f"Error: Could not create lock file at {config.LOCK_FILE}: {e}"
                )
                raise

        def read(self) -> int | None:
            """Reads the process ID from the lock file."""
            try:
                with open(config.LOCK_FILE, "r") as f:
                    return int(f.read().strip())
            except (FileNotFoundError, ValueError):
                return None

        def remove(self) -> None:
            """Removes the training lock file if it exists."""
            if os.path.exists(config.LOCK_FILE):
                try:
                    os.remove(config.LOCK_FILE)
                except OSError as e:
                    print(
                        f"Error: Could not remove lock file at {config.LOCK_FILE}: {e}"
                    )

        def is_locked(self) -> bool:
            """Checks if the training lock file exists."""
            return os.path.exists(config.LOCK_FILE)

    class StatusFileManager:
        """Manages the status log file."""

        def write(self, message: str) -> None:
            """Writes a message to the status log file."""
            try:
                with open(config.STATUS_LOG, "w") as f:
                    f.write(message)
            except OSError as e:
                print(f"Error writing to status file: {e}")

        def read(self) -> str:
            """Reads the current status from the status file."""
            try:
                with open(config.STATUS_LOG, "r") as f:
                    return f.read().strip()
            except FileNotFoundError:
                return "Initializing"

        def remove(self) -> None:
            """Removes the status log file."""
            if os.path.exists(config.STATUS_LOG):
                os.remove(config.STATUS_LOG)

    class LogFileManager:
        """Manages the training stdout/stderr log files."""

        def __init__(self):
            self.stdout_handle = None
            self.stderr_handle = None

        def open(self) -> tuple:
            """Opens training log files and returns the file handles."""
            try:
                self.stdout_handle = open(config.TRAINER_STDOUT_LOG, "w")
                self.stderr_handle = open(config.TRAINER_STDERR_LOG, "w")
                return self.stdout_handle, self.stderr_handle
            except OSError as e:
                print(f"Error opening log files: {e}")
                if self.stdout_handle:
                    self.stdout_handle.close()
                raise

        def close(self) -> None:
            """Closes any open log file handles."""
            if self.stdout_handle and not self.stdout_handle.closed:
                self.stdout_handle.close()
                self.stdout_handle = None
            if self.stderr_handle and not self.stderr_handle.closed:
                self.stderr_handle.close()
                self.stderr_handle = None

        def read_stderr(self) -> str:
            """Reads the content of the standard error log file."""
            try:
                with open(config.TRAINER_STDERR_LOG, "r") as f:
                    return f.read()
            except FileNotFoundError:
                return ""

        def remove(self) -> None:
            """Removes the training log files."""
            self.close()
            if os.path.exists(config.TRAINER_STDOUT_LOG):
                os.remove(config.TRAINER_STDOUT_LOG)
            if os.path.exists(config.TRAINER_STDERR_LOG):
                os.remove(config.TRAINER_STDERR_LOG)
                
    class TensorBoardFileManager:
        """Manages finding and cleaning up TensorBoard event files."""

        def find_latest_event_file(self, since_time: float = 0) -> str | None:
            """Finds the latest event file in the log directory."""
            log_dir = config.TENSORBOARD_LOGDIR
            event_files = []
            for root, _, files in os.walk(log_dir):
                for file in files:
                    if "events.out.tfevents" in file:
                        file_path = os.path.join(root, file)
                        event_files.append(
                            (
                                file_path,
                                os.path.getctime(file_path),
                                os.path.getmtime(file_path),
                            )
                        )
            
            if not event_files:
                return None

            # Sort by modification time (newest first)
            event_files.sort(key=lambda x: x[2], reverse=True)

            if not since_time:
                return event_files[0][0]

            # Only consider files created after the specified time
            relevant_files = [
                file_path
                for file_path, created_time, _ in event_files
                if created_time >= since_time
            ]
            return relevant_files[0] if relevant_files else None

        def remove(self) -> None:
            """Placeholder for cleaning up tensorboard files if needed."""
            # For now, we don't delete the tensorboard logs automatically.
            # This can be implemented later if required.
            pass
        
