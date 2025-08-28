from typing import Any

from gemma import gm, peft


class Model:
    """Class for creating model instances."""

    @staticmethod
    def create_standard_model(model_variant: str) -> Any:
        """Create the standard model instance."""
        # if model_variant in ["Gemma3n_E2B", "Gemma3n_E4B"]:
        #     model_class = getattr(gm.nn.gemma3n, model_variant)
        model_class = getattr(gm.nn, model_variant)
        return model_class(tokens="batch.input")

    @staticmethod
    def create_lora_model(model_variant: str, lora_rank: int) -> gm.nn.LoRA:
        """Create the LoRA model instance."""
        base_model = Model.create_standard_model(model_variant)
        return gm.nn.LoRA(rank=lora_rank, model=base_model)

    @staticmethod
    def create_quantization_aware_model(model_variant: str) -> Any:
        """Create the quantization aware model instance."""
        base_model = Model.create_standard_model(model_variant)
        return gm.nn.QuantizationAwareWrapper(
            method=peft.QuantizationMethod.INT8,
            model=base_model,
        )

    @staticmethod
    def create_quantization_aware_model_inference(model_variant: str) -> Any:
        """Create the quantization aware model instance."""
        base_model = Model.create_standard_model(model_variant)
        return gm.nn.IntWrapper(model=base_model)

    @staticmethod
    def load_trained_params(checkpoint_path: str, method: str) -> Any:
        """Load trained parameters from a specific checkpoint path."""
        params = gm.ckpts.load_params(checkpoint_path)
        if method == "QuantizationAware":
            return peft.quantize(params, method=peft.QuantizationMethod.INT8)
        return params
