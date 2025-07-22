import os
from enum import Enum


class TrainingStatus(Enum):
    """Training status enum."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    ORPHANED = "ORPHANED"

class AppConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
        return cls._instance


    # --- Training process constants ---
    TRAINING_STATE_FILE = ".training_state.json"
    CHECKPOINT_FOLDER = os.path.abspath("./checkpoints/")
    TRAINER_STDOUT_LOG = "trainer_stdout.log"
    TRAINER_STDERR_LOG = "trainer_stderr.log"
    TRAINER_MAIN_PATH = "backend/trainer_main.py"

    # --- Model configuration ---
    MODEL_OPTIONS = [
        "Gemma2_2B",
        "Gemma2_9B",
        "Gemma2_27B",
        "Gemma3_1B",
        "Gemma3_4B",
        "Gemma3_12B",
        "Gemma3_27B",
    ]


def get_config():
    """Returns a singleton instance of the AppConfig."""
    return AppConfig()
