from typing import Any, Optional, Tuple
import os
from pathlib import Path
import optax
from kauldron import kd
import sys
import traceback

from config.training_config import (
    ModelConfig,
    DataConfig,
    LOCK_FILE,
    MODEL_ARTIFACT,
)
from backend.utils.manager import StatusManager
from backend.core.model import ModelFactory
from backend.core.loss import LossFactory
from backend.core.sampler import SamplerFactory
from data_pipeline import create_pipeline


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
            workdir="/tmp/ckpts",
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

    def train(self) -> Tuple[Any, Any, Any]:
        """Execute the training process."""
        state, aux = None, None
        try:
            Path(LOCK_FILE).touch()
            self.status_manager.update("Initializing training...")

            self.setup_environment()
            self.pipeline = create_pipeline(self.data_config.__dict__)
            train_ds = self.pipeline.get_train_dataset()
            self.model = ModelFactory.create_model(self.model_config)
            self.trainer = self.create_trainer(self.model, train_ds)

            self.status_manager.update("Starting training...")
            state, aux = self.trainer.train()

            self.status_manager.update("Creating sampler...")
            sampler = SamplerFactory.create_sampler(
                self.model, state, self.pipeline.tokenizer
            )

            self.status_manager.update("Saving model...")
            Path(MODEL_ARTIFACT).touch()
            self.status_manager.update("Training completed successfully!")

            return state, aux, sampler

        except Exception:
            error_traceback_str = traceback.format_exc()
            print(error_traceback_str, file=sys.stderr)
            error_summary = error_traceback_str.strip().split("\n")[-1]
            self.status_manager.update(f"Error: {error_summary}")
            raise
        finally:
            if os.path.exists(LOCK_FILE):
                os.remove(LOCK_FILE)
            self.status_manager.cleanup()
