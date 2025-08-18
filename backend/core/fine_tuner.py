from typing import Any
from kauldron import kd

from backend.core.loss import Loss
from backend.core.model import Model
from backend.core.optimizer import Optimizer
from backend.core.checkpoint import Checkpoint
from config.dataclass import TrainingConfig

DEFAULT_SEED = 42
LOG_METRICS_INTERVAL = 1
LOG_SUMMARIES_INTERVAL = 1000


class Trainer:
    """Base class for trainers."""

    def create_trainer(
        self,
        model: Any,
        init_transform: Any,
        optimizer: Any,
        train_ds: kd.data.Pipeline,
        workdir: str,
        num_train_steps: int,
    ) -> kd.train.Trainer:
        trainer = kd.train.Trainer(
            seed=42,
            workdir=workdir,
            train_ds=train_ds,
            model=model,
            init_transform=init_transform,
            num_train_steps=num_train_steps,
            train_losses={"loss": Loss.create_loss()},
            optimizer=optimizer,
            log_metrics_every=LOG_METRICS_INTERVAL,
            log_summaries_every=LOG_SUMMARIES_INTERVAL,
            checkpointer=kd.ckpts.Checkpointer(
                save_interval_steps=num_train_steps
            ),
        )
        return trainer


class StandardTrainer(Trainer):
    """Trainer for standard models."""

    def create_trainer(
        self,
        training_config: TrainingConfig,
        train_ds: kd.data.Pipeline,
        workdir: str,
    ) -> kd.train.Trainer:
        model = Model.create_standard_model(
            training_config.model_config.model_variant
        )
        init_transform = Checkpoint.create_standard_checkpoint(
            training_config.model_config.model_variant
        )
        optimizer = Optimizer.create_standard_optimizer(
            training_config.model_config.learning_rate
        )
        return super().create_trainer(
            model,
            init_transform,
            optimizer,
            train_ds,
            workdir,
            training_config.model_config.epochs,
        )


class LoRATrainer(Trainer):
    """Trainer for LoRA models."""

    def create_trainer(
        self,
        training_config: TrainingConfig,
        train_ds: kd.data.Pipeline,
        workdir: str,
    ) -> kd.train.Trainer:
        model = Model.create_lora_model(
            training_config.model_config.model_variant,
            training_config.model_config.parameters.lora_rank,
        )
        init_transform = Checkpoint.create_lora_checkpoint(
            training_config.model_config.model_variant
        )
        optimizer = Optimizer.create_lora_optimizer(
            training_config.model_config.learning_rate
        )
        return super().create_trainer(
            model,
            init_transform,
            optimizer,
            train_ds,
            workdir,
            training_config.model_config.epochs,
        )


class QuantizationAwareTrainer(Trainer):
    """Trainer for quantization aware models."""

    def create_trainer(
        self,
        training_config: TrainingConfig,
        train_ds: kd.data.Pipeline,
        workdir: str,
    ) -> kd.train.Trainer:
        model = Model.create_quantization_aware_model(
            training_config.model_config.model_variant
        )
        init_transform = Checkpoint.create_standard_checkpoint(
            training_config.model_config.model_variant
        )
        optimizer = Optimizer.create_standard_optimizer(
            training_config.model_config.learning_rate
        )
        return super().create_trainer(
            model,
            init_transform,
            optimizer,
            train_ds,
            workdir,
            training_config.model_config.epochs,
        )
FINE_TUNE_STRATEGIES: dict[str, Trainer] = {
    "Standard": StandardTrainer(),
    "LoRA": LoRATrainer(),
    "QuantizationAware": QuantizationAwareTrainer(),
}
