import os
import sys
import time
import traceback
from typing import Any, Optional, Tuple

import optax
from data_pipeline import create_pipeline
from kauldron import kd

from config.dataclass import TrainingConfig
from config.app_config import get_config
from backend.core.fine_tuner import FINE_TUNE_STRATEGIES

config = get_config()


class ModelTrainer:
    """Main class for handling the model training process."""

    def __init__(self, training_config: TrainingConfig, work_dir: str):
        self.training_config = training_config
        self.pipeline = None
        self.trainer = None
        self.workdir = work_dir

    def setup_environment(self) -> None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"


    def train(self) -> Tuple[Any, Any]:
        """Execute the training process."""
        state, aux = None, None
        try:
            self.setup_environment()
            pipeline = create_pipeline(self.training_config.data_config)
            train_ds = pipeline.get_train_dataset()
            trainer = FINE_TUNE_STRATEGIES[
                self.training_config.model_config.method
            ].create_trainer(self.training_config, train_ds, self.workdir)
            state, aux = trainer.train()
            return state, aux

        except Exception:
            error_traceback_str = traceback.format_exc()
            print(error_traceback_str, file=sys.stderr)
            error_summary = error_traceback_str.strip().split("\n")[-1]
            raise error_summary
