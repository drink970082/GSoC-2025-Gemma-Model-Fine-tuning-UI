import os
import sys
import time
import traceback
from typing import Any, Optional, Tuple

import optax
from data_pipeline import create_pipeline
from kauldron import kd
from orbax import checkpoint as ocp

from backend.core.loss import LossFactory
from backend.core.model import ModelFactory
from backend.manager.status_manager import StatusManager
from config.app_config import DataConfig, ModelConfig, get_config

config = get_config()


class ModelTrainer:
    """Main class for handling the model training process."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        work_dir: str,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.status_manager = StatusManager()
        self.pipeline = None
        self.model = None
        self.trainer = None
        self.workdir = work_dir

    def setup_environment(self) -> None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

    def create_trainer(self, model: Any, train_ds: Any) -> kd.train.Trainer:
        checkpointer = kd.ckpts.Checkpointer(
            save_interval_steps=100,
            save_on_steps=[
                self.model_config.epochs
            ],  # Explicitly save at final step
            max_to_keep=3,
        )
        trainer = kd.train.Trainer(
            seed=42,
            workdir=self.workdir,
            train_ds=train_ds,
            model=model,
            init_transform=ModelFactory.create_checkpoint(self.model_config),
            num_train_steps=self.model_config.epochs,
            train_losses={"loss": LossFactory.create_loss()},
            optimizer=optax.adafactor(
                learning_rate=self.model_config.learning_rate
            ),
            log_metrics_every=1,
            log_summaries_every=1000,
            checkpointer=checkpointer,
        )
        return trainer

    def train(self) -> Tuple[Any, Any]:
        """Execute the training process."""
        self.status_manager.set_work_dir(self.workdir)
        state, aux = None, None
        try:
            self.status_manager.update("Initializing training...")
            self.setup_environment()
            # Create pipeline and dataset
            pipeline = create_pipeline(self.data_config.__dict__)
            train_ds = pipeline.get_train_dataset()
            # Create model
            model = ModelFactory.create_model(self.model_config)
            # Create trainer and checkpointer
            trainer = self.create_trainer(model, train_ds)
            self.status_manager.update("Training in progress...")
            state, aux = trainer.train()
            self.status_manager.update("Training completed.")
            return state, aux

        except Exception:
            error_traceback_str = traceback.format_exc()
            print(error_traceback_str, file=sys.stderr)
            error_summary = error_traceback_str.strip().split("\n")[-1]
            self.status_manager.update(f"Error: {error_summary}")
            raise
