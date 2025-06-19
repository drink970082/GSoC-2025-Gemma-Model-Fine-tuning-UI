import os
from typing import Optional
from config.training_config import DEFAULT_MODEL_CONFIG, MODEL_ARTIFACT
from backend.core.model import ModelFactory
from gemma import gm
from config.training_config import ModelConfig


class InferenceService:
    """Service for running inference with trained models."""

    def __init__(self, model_config: ModelConfig):
        self.model_config: ModelConfig = model_config
        self.model = None
        self.params = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> bool:
        """Load the trained model if available."""
        if not os.path.exists(MODEL_ARTIFACT):
            return False

        # Load trained parameters
        self.params = ModelFactory.load_trained_params()

        # Create model instance
        self.model = ModelFactory.create_model(self.model_config)

        # Create tokenizer
        self.tokenizer = gm.text.Gemma3Tokenizer()

        self.sampler = gm.text.ChatSampler(
            model=self.model,
            params=self.params,
            tokenizer=self.tokenizer,
        )

        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return self.sampler.chat(prompt)
