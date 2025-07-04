import os
from pathlib import Path
import shutil
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

    def list_checkpoints(self) -> list[str]:
        """Return a list of checkpoint directory names, sorted by creation time (newest first)."""
        if not os.path.exists(self.work_dir):
            return []
        subdirs = [p for p in Path(self.work_dir).iterdir() if p.is_dir()]
        subdirs.sort(key=lambda p: p.stat().st_ctime, reverse=True)
        return [p.name for p in subdirs]

    def get_latest_checkpoint(self) -> str:
        """Return the path of the most recently created checkpoint directory."""
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        return checkpoints[0]

    def delete_checkpoint(self, checkpoint_name):
        """Delete the specified checkpoint directory."""
        checkpoint_path = Path(self.work_dir) / checkpoint_name
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            shutil.rmtree(checkpoint_path)
            return True
        return False

    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """Load the most recent trained model if available."""
        self._loaded = False
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if not checkpoint_path:
                return False
        else:
            if not os.path.exists(checkpoint_path):
                return False

        # Load trained parameters from the latest checkpoint
        self.params = Model.load_trained_params(checkpoint_path)

        if self.training_config.model_config.method == "LoRA":
            self.model = Model.create_lora_model(
                self.training_config.model_config.model_variant,
                self.training_config.model_config.parameters.lora_rank,
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
