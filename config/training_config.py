import os
from pathlib import Path

# Training process constants
LOCK_FILE = ".training.lock"
STATUS_LOG = "status.log"
MODEL_ARTIFACT = os.path.abspath("./test_checkpoints/")  # ← Convert to absolute
# Tensorboard log directory
TENSORBOARD_LOGDIR = "/tmp/ckpts"
TENSORBOARD_PORT = 6007
TRAINER_STDOUT_LOG = "trainer_stdout.log"
TRAINER_STDERR_LOG = "trainer_stderr.log"
TRAINER_MAIN_PATH = "backend/trainer_main.py"

# config/training_config.py
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the model training."""

    model_variant: str
    epochs: int
    learning_rate: float
    batch_size: int


@dataclass
class DataConfig:
    """Configuration for the data pipeline."""

    source: str
    dataset_name: str
    split: str
    batch_size: int
    shuffle: bool
    seq2seq_in_prompt: str
    seq2seq_in_response: str
    seq2seq_max_length: int
    seq2seq_truncate: bool


# Default configurations
DEFAULT_DATA_CONFIG = {
    "source": "tensorflow",
    "dataset_name": "mtnt",
    "split": "train",
    "batch_size": 4,
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
