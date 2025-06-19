import os
from pathlib import Path
from typing import Optional

from gemma import gm

from backend.core.model import ModelFactory
from config.training_config import CHECKPOINT_FOLDER, ModelConfig


class Inferencer:
    """Service for running inference with trained models."""

    def __init__(self, model_config: ModelConfig):
        self.model_config: ModelConfig = model_config
        self.model = None
        self.params = None
        self.tokenizer = None
        self._loaded = False

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recently created checkpoint directory."""
        if not os.path.exists(CHECKPOINT_FOLDER):
            return None
        subdirs = [p for p in Path(CHECKPOINT_FOLDER).iterdir()]
        if not subdirs:
            return None

        # Return the path of the most recently created directory
        latest_subdir = max(subdirs, key=lambda p: p.stat().st_ctime)
        return str(latest_subdir)

    def load_model(self) -> bool:
        """Load the most recent trained model if available."""
        latest_checkpoint_path = self._find_latest_checkpoint()
        if not latest_checkpoint_path:
            return False

        # Load trained parameters from the latest checkpoint
        self.params = ModelFactory.load_trained_params(latest_checkpoint_path)

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
