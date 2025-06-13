from typing import Any
from gemma import gm
from config.training_config import ModelConfig


class ModelFactory:
    """Factory for creating model instances."""

    @staticmethod
    def create_model(config: ModelConfig) -> Any:
        """Create the model instance."""
        model_class = getattr(gm.nn, config.model_variant)
        return model_class(tokens="batch.input")

    @staticmethod
    def create_checkpoint(config: ModelConfig) -> Any:
        """Create the model checkpoint."""
        checkpoint_path = getattr(
            gm.ckpts.CheckpointPath, (config.model_variant + "_IT").upper()
        )
        return gm.ckpts.LoadCheckpoint(path=checkpoint_path)
