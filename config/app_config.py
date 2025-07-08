import os

from enum import Enum, auto


class TrainingStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()
    ORPHANED = auto()


class AppConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
        return cls._instance

    # --- default values ---
    # Default configurations
    DEFAULT_DATA_CONFIG = {
        "source": "tensorflow",
        "dataset_name": "mtnt",
        "split": "train",
        "shuffle": True,
        "seq2seq_in_prompt": "src",
        "seq2seq_max_length": 200,
        "seq2seq_in_response": "dst",
        "seq2seq_truncate": True,
    }

    DEFAULT_MODEL_CONFIG = {
        "model_variant": "Gemma3_1B",
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-3,
    }

    # --- Training process constants ---
    LOCK_FILE = ".training.lock"
    STATUS_LOG = "status.log"
    CHECKPOINT_FOLDER = os.path.abspath("./checkpoints/")
    TENSORBOARD_LOGDIR = "checkpoints/"
    TENSORBOARD_PORT = 6007
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
    """
    Returns a singleton instance of the AppConfig.
    """
    return AppConfig()
