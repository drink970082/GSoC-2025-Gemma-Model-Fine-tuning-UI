from typing import Any

from gemma import gm


class Model:
    """Class for creating model instances."""

    @staticmethod
    def create_standard_model(model_variant: str) -> Any:
        """Create the model instance."""
        model_class = getattr(gm.nn, model_variant)
        return model_class(tokens="batch.input")

    @staticmethod
    def create_lora_model(model_variant: str, lora_rank: int) -> Any:
        """Create the model instance."""
        base_model = Model.create_standard_model(model_variant)
        return gm.nn.LoRA(rank=lora_rank, model=base_model)

    @staticmethod
    def create_standard_checkpoint(model_variant: str) -> Any:
        """Create the model checkpoint."""
        checkpoint_path = getattr(
            gm.ckpts.CheckpointPath, (model_variant + "_IT").upper()
        )
        return gm.ckpts.LoadCheckpoint(path=checkpoint_path)

    @staticmethod
    def create_lora_checkpoint(model_variant: str, lora_rank: int) -> Any:
        """Create the model checkpoint."""
        base_checkpoint = Model.create_standard_checkpoint(model_variant)
        return gm.ckpts.SkipLoRA(
            wrapped=base_checkpoint,
        )

    @staticmethod
    def load_trained_params(checkpoint_path: str):
        """Load trained parameters from a specific checkpoint path."""
        return gm.ckpts.load_params(checkpoint_path)
