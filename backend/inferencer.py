import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple, List

from gemma import gm

from backend.core.model import Model
from config.app_config import get_config
from config.dataclass import DpoParams, LoraParams, ModelConfig, TrainingConfig

config = get_config()

# Constants
CHECKPOINT_SUBDIR = "checkpoints"
MODEL_CONFIG_FILE = "model_config.json"


class Inferencer:
    """Service for running inference with trained models."""

    def __init__(self, work_dir: Optional[str] = None) -> None:
        self.model: Optional[Any] = None
        self.params: Optional[Any] = None
        self.tokenizer: Optional[gm.text.Gemma3Tokenizer] = None
        self.sampler: Optional[gm.text.ChatSampler] = None
        self._loaded: bool = False
        self.work_dir: str = work_dir or config.CHECKPOINT_FOLDER

    def list_checkpoints(self) -> List[str]:
        """Return a list of checkpoint directory names, sorted by creation time (newest first)."""
        if not os.path.exists(self.work_dir):
            return []

        subdirs = [p for p in Path(self.work_dir).iterdir() if p.is_dir()]
        subdirs.sort(key=lambda p: p.stat().st_ctime, reverse=True)
        return [p.name for p in subdirs]

    def get_latest_checkpoint(self) -> Optional[str]:
        """Return the path of the most recently created checkpoint directory."""
        checkpoints = self.list_checkpoints()
        print(checkpoints)
        return checkpoints[0] if checkpoints else None

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete the specified checkpoint directory."""
        checkpoint_path = Path(self.work_dir) / checkpoint_name
        if checkpoint_path.exists() and checkpoint_path.is_dir():
            shutil.rmtree(checkpoint_path)
            return True
        return False


    def load_model(self, checkpoint_path: Optional[str] = None) -> bool:
        """Load the most recent trained model if available."""
        self.clear_model()

        # Get checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if not checkpoint_path:
                return False

        full_checkpoint_path = str(Path(self.work_dir) / checkpoint_path)
        if not os.path.exists(full_checkpoint_path):
            return False

        try:
            # Load model configuration
            config_file = os.path.join(full_checkpoint_path, MODEL_CONFIG_FILE)
            with open(config_file, "r") as f:
                model_config = self._parse_model_config(json.load(f))

            # Load trained parameters
            self.params = Model.load_trained_params(
                self._get_checkpoint_path(full_checkpoint_path), method=model_config.method
            )

            # Create model
            self.model = self._create_model_from_config(model_config)

            # Create tokenizer and sampler
            self.tokenizer = gm.text.Gemma3Tokenizer()
            self.sampler = gm.text.ChatSampler(
                model=self.model, params=self.params, tokenizer=self.tokenizer
            )

            self._loaded = True
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.clear_model()
            return False

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        return self.sampler.chat(prompt)

    def get_tokenizer(self) -> Optional[gm.text.Gemma3Tokenizer]:
        """Get the tokenizer if loaded."""
        return self.tokenizer if self._loaded else None

    def get_sampler(self) -> Optional[gm.text.ChatSampler]:
        """Get the sampler if loaded."""
        return self.sampler if self._loaded else None

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the loaded tokenizer."""
        if not self._loaded or not self.tokenizer:
            return 0
        return len(self.tokenizer.encode(text))

    def clear_model(self) -> None:
        """Clear all model-related attributes."""
        self.model = None
        self.params = None
        self.tokenizer = None
        self.sampler = None
        self._loaded = False
        
    def _get_checkpoint_path(self, work_dir: str) -> str:
        """Get the path to the checkpoint subdirectory."""
        checkpoint_path = os.path.join(work_dir, CHECKPOINT_SUBDIR)
        subdirs = [p for p in Path(checkpoint_path).iterdir() if p.is_dir()]
        return subdirs[0]

    def _parse_model_config(self, model_config_dict: dict) -> ModelConfig:
        """Parse model configuration from dictionary."""
        parameters = None
        if model_config_dict.get("parameters"):
            method = model_config_dict["method"]
            if method == "LoRA":
                parameters = LoraParams(**model_config_dict["parameters"])
        return ModelConfig(
            model_variant=model_config_dict["model_variant"],
            epochs=model_config_dict["epochs"],
            learning_rate=model_config_dict["learning_rate"],
            method=model_config_dict["method"],
            parameters=parameters,
        )

    def _create_model_from_config(self, model_config: ModelConfig) -> Any:
        """Create model instance based on configuration."""
        if model_config.method == "LoRA":
            return Model.create_lora_model(
                model_config.model_variant, model_config.parameters.lora_rank
            )
        elif model_config.method == "QuantizationAware":
            return Model.create_quantization_aware_model_inference(model_config.model_variant)
        else:
            return Model.create_standard_model(model_config.model_variant)

