from typing import Any

from gemma import gm


class Model:
    """Class for creating model instances."""

    @staticmethod
    def create_standard_model(model_variant: str) -> Any:
        """Create the standard model instance."""
        model_class = getattr(gm.nn, model_variant)
        return model_class(tokens="batch.input")

    @staticmethod
    def create_lora_model(model_variant: str, lora_rank: int) -> gm.nn.LoRA:
        """Create the LoRA model instance."""
        base_model = Model.create_standard_model(model_variant)
        return gm.nn.LoRA(rank=lora_rank, model=base_model)

    @staticmethod
    def load_trained_params(checkpoint_path: str) -> Any:
        """Load trained parameters from a specific checkpoint path."""
        print(checkpoint_path)
        return gm.ckpts.load_params(checkpoint_path)
