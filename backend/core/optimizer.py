from typing import Any
import optax
from kauldron import kd


class Optimizer:
    """Factory for creating optimizers."""

    @staticmethod
    def create_standard_optimizer(learning_rate: float) -> Any:
        return optax.adafactor(learning_rate=learning_rate)

    @staticmethod
    def create_lora_optimizer(learning_rate: float) -> Any:
        standard_optimizer = Optimizer.create_standard_optimizer(learning_rate)
        return kd.optim.partial_updates(
            standard_optimizer,
            mask=kd.optim.select("lora"),
        )
