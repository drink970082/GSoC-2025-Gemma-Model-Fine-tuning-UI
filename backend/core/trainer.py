import os
import sys
import traceback
from typing import Any, Optional, Tuple

from config.dataclass import TrainingConfig
from config.app_config import get_config
from backend.data_pipeline import create_pipeline, DataPipeline
from backend.core.fine_tuner import FINE_TUNE_STRATEGIES

config = get_config()

XLA_MEM_FRACTION = "1.00"

class ModelTrainer:
    """Main class for handling the model training process."""

    def __init__(self, training_config: TrainingConfig, work_dir: str) -> None:
        """Initialize the ModelTrainer."""
        self.training_config: TrainingConfig = training_config
        self.workdir: str = work_dir

    def setup_environment(self) -> None:
        """Set up the environment for the training process."""
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = XLA_MEM_FRACTION


    def train(self) -> Tuple[Any, Any]:
        """Execute the training process."""
        try:
            self.setup_environment()
            pipeline = create_pipeline(self.training_config.data_config)
            train_ds = pipeline.get_train_dataset()
            trainer = FINE_TUNE_STRATEGIES[
                self.training_config.model_config.method
            ].create_trainer(self.training_config, train_ds, self.workdir)
            return trainer.train()

        except Exception as e:
            error_traceback_str = traceback.format_exc()
            print(error_traceback_str, file=sys.stderr)
            raise e
