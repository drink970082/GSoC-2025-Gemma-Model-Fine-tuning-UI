import os
import pickle
from typing import Any

from gemma import gm
from config.training_config import MODEL_ARTIFACT, ModelConfig


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
    def load_trained_params() -> Any:
        """Load trained parameters from saved checkpoint."""
        if not os.path.exists(MODEL_ARTIFACT):
            raise FileNotFoundError(
                f"No trained parameters found at {MODEL_ARTIFACT}"
            )

        return gm.ckpts.load_params(MODEL_ARTIFACT)
