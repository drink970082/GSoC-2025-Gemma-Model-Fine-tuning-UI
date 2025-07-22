import optax
from kauldron import kd


class Optimizer:
    """Class for creating optimizers."""

    @staticmethod
    def create_standard_optimizer(lr: float) -> optax.GradientTransformation:
        """Create the standard Adafactor optimizer."""
        return optax.adafactor(learning_rate=lr)

    @staticmethod
    def create_lora_optimizer(lr: float) -> optax.GradientTransformation:
        """Create the LoRA optimizer (partial updates)."""
        standard_optimizer = Optimizer.create_standard_optimizer(lr)
        return kd.optim.partial_updates(
            standard_optimizer, mask=kd.optim.select("lora")
        )
