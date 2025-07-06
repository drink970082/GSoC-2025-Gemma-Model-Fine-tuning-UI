from backend.core.model import Model
from config.dataclass import LoraParams
from kauldron import kd
from backend.core.loss import Loss
from backend.core.model import Model
from backend.core.optimizer import Optimizer

from config.dataclass import TrainingConfig


class Trainer:
    def create_trainer(
        self,
        model,
        init_transform,
        optimizer,
        train_ds,
        workdir,
        num_train_steps,
    ) -> kd.train.Trainer:
        checkpointer = kd.ckpts.Checkpointer(
            save_interval_steps=100,
            save_on_steps=[num_train_steps],  # Explicitly save at final step
        )
        trainer = kd.train.Trainer(
            seed=42,
            workdir=workdir,
            train_ds=train_ds,
            model=model,
            init_transform=init_transform,
            num_train_steps=num_train_steps,
            train_losses={"loss": Loss.create_loss()},
            optimizer=optimizer,
            log_metrics_every=1,
            log_summaries_every=1000,
            checkpointer=checkpointer,
        )
        return trainer


class StandardTrainer(Trainer):
    def create_trainer(
        self, training_config: TrainingConfig, train_ds, workdir
    ) -> kd.train.Trainer:
        model = Model.create_standard_model(
            training_config.model_config.model_variant
        )
        init_transform = Model.create_standard_checkpoint(
            training_config.model_config
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
    def create_trainer(
        self, training_config: TrainingConfig, train_ds, workdir
    ) -> kd.train.Trainer:
        model = Model.create_lora_model(
            training_config.model_config.model_variant,
            training_config.method_config.parameters.lora_rank,
        )
        init_transform = Model.create_lora_checkpoint(
            training_config.model_config.model_variant,
            training_config.method_config.parameters.lora_rank,
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


FINE_TUNE_STRATEGIES: dict[str, Trainer] = {
    "Standard": StandardTrainer(),
    "LoRA": LoRATrainer(),
}
