import os
from pathlib import Path
from typing import Optional

from gemma import gm

from backend.core.model import Model
from config.app_config import get_config
from config.dataclass import TrainingConfig

config = get_config()


class Inferencer:
    """Service for running inference with trained models."""

    def __init__(self, training_config: TrainingConfig, work_dir: str):
        self.training_config: TrainingConfig = training_config
        self.model = None
        self.params = None
        self.tokenizer = None
        self._loaded = False
        self.work_dir = work_dir

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recently created checkpoint directory."""
        if not os.path.exists(self.work_dir):
            return None
        subdirs = [p for p in Path(self.work_dir).iterdir()]
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
        self.params = Model.load_trained_params(latest_checkpoint_path)

        if self.training_config.method_config.name == "LoRA":
            self.model = Model.create_lora_model(
                self.training_config.model_config.model_variant,
                self.training_config.method_config.parameters.lora_rank,
            )
        else:
            self.model = Model.create_standard_model(
                self.training_config.model_config.model_variant
            )

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
