import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple
import time

import optax
from data_pipeline import create_pipeline
from kauldron import kd
from orbax import checkpoint as ocp

from backend.core.loss import LossFactory
from backend.core.model import ModelFactory
from backend.manager.status_manager import StatusManager
from config.app_config import get_config, DataConfig, ModelConfig

config = get_config()


class ModelTrainer:
    """Main class for handling the model training process."""

    def __init__(
        self,
        data_config: DataConfig,
        model_config: ModelConfig,
        status_manager: Optional[StatusManager] = None,
    ):
        self.data_config = data_config
        self.model_config = model_config
        self.status_manager = status_manager or StatusManager()
        self.pipeline = None
        self.model = None
        self.trainer = None

    def setup_environment(self) -> None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

    def create_trainer(self, model: Any, train_ds: Any) -> kd.train.Trainer:
        return kd.train.Trainer(
            seed=42,
            workdir="/tmp/ckpts/",
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
        )

    def train(self) -> Tuple[Any, Any]:
        """Execute the training process."""
        state, aux = None, None
        try:
            Path(config.LOCK_FILE).touch()
            self.status_manager.update("Initializing training...")

            self.setup_environment()
            self.pipeline = create_pipeline(self.data_config.__dict__)
            train_ds = self.pipeline.get_train_dataset()
            self.model = ModelFactory.create_model(self.model_config)
            self.trainer = self.create_trainer(self.model, train_ds)

            self.status_manager.update("Starting training...")
            state, aux = self.trainer.train()
            self.status_manager.update("Saving model...")
            ckpt = ocp.StandardCheckpointer()
            ckpt.save(
                os.path.abspath(f"{config.CHECKPOINT_FOLDER}/{time.time()}"),
                state.params,
            )
            ckpt.wait_until_finished()
            return state, aux

        except Exception:
            error_traceback_str = traceback.format_exc()
            print(error_traceback_str, file=sys.stderr)
            error_summary = error_traceback_str.strip().split("\n")[-1]
            self.status_manager.update(f"Error: {error_summary}")
            raise
        finally:
            if os.path.exists(config.LOCK_FILE):
                os.remove(config.LOCK_FILE)
            self.status_manager.cleanup()
