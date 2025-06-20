from typing import Any

from gemma import gm

from config.app_config import ModelConfig


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

    @staticmethod
    def load_trained_params(checkpoint_path: str):
        """Load trained parameters from a specific checkpoint path."""
        return gm.ckpts.load_params(checkpoint_path)
