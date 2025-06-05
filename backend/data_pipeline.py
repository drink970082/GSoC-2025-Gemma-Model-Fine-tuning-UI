import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import atexit
import optax
import pandas as pd
import treescope
from gemma import gm
from kauldron import kd


class DataPipeline:
    """Base class for data pipelines."""

    def __init__(self, config: Dict):
        """Initialize the data pipeline with configuration."""
        self.config = config
        self.data = None
        self.processed_data = None
        self.tokenizer = gm.text.Gemma3Tokenizer()

    def load_data(self) -> None:
        """Load data from the source."""
        raise NotImplementedError

    def preprocess(self) -> None:
        """Preprocess the loaded data."""
        pass

    def get_dataset(self) -> Dict:
        """Return the processed dataset."""
        return self.data


class JSONPipeline(DataPipeline):
    """Pipeline for JSON format data."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.temp_file_path = self.config.get("data_path")
        if self.temp_file_path:
            atexit.register(self.cleanup)

    def load_data(self) -> None:
        """Load data from JSON file."""
        try:
            # Use Json loader with file path
            ds = kd.data.py.Json(
                path=self.config["data_path"],
                shuffle=self.config.get("shuffle", True),
                batch_size=8,
                transforms=[
                    # Create the model inputs/targets/loss_mask.
                    gm.data.Seq2SeqTask(
                        # Select which field from the dataset to use.
                        in_prompt=self.config["seq2seq_in_prompt"],
                        in_response=self.config["seq2seq_in_response"],
                        # Output batch is {'input': ..., 'target': ..., 'loss_mask': ...}
                        out_input="input",
                        out_target="target",
                        out_target_mask="loss_mask",
                        tokenizer=self.tokenizer,
                        # Padding parameters
                        max_length=self.config["seq2seq_max_length"],
                        truncate=self.config["seq2seq_truncate"],
                    ),
                ],
            )
            self.data = ds
        except Exception as e:
            raise ValueError(f"Failed to load JSON data: {str(e)}")

    def cleanup(self):
        """
        Deletes the temporary file.
        This method is now automatically called by Python upon script termination.
        """
        # Check for the path one last time, as it might have been cleaned already
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)


class TensorFlowPipeline(DataPipeline):
    """Pipeline for TensorFlow datasets."""

    def load_data(self) -> None:
        """Load data from TensorFlow dataset."""
        try:
            ds = kd.data.py.Tfds(
                name=self.config["dataset_name"],
                split=self.config.get("split") or "train",
                shuffle=self.config.get("shuffle", True),
                batch_size=8,
                transforms=[
                    # Create the model inputs/targets/loss_mask.
                    gm.data.Seq2SeqTask(
                        # Select which field from the dataset to use.
                        in_prompt=self.config["seq2seq_in_prompt"],
                        in_response=self.config["seq2seq_in_response"],
                        # Output batch is {'input': ..., 'target': ..., 'loss_mask': ...}
                        out_input="input",
                        out_target="target",
                        out_target_mask="loss_mask",
                        tokenizer=self.tokenizer,
                        # Padding parameters
                        max_length=self.config["seq2seq_max_length"],
                        truncate=self.config["seq2seq_truncate"],
                    ),
                ],
            )
            self.data = ds
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow dataset: {str(e)}")


class HuggingFacePipeline(DataPipeline):
    """Pipeline for HuggingFace datasets."""

    def load_data(self) -> None:
        """Load data from HuggingFace dataset."""
        try:
            ds = kd.data.py.HuggingFace(
                path=self.config["dataset_name"],
                config=self.config["dataset_config"] or "main",
                split=self.config.get("split") or "train",
                shuffle=self.config.get("shuffle", True),
                batch_size=8,
                transforms=[
                    # Create the model inputs/targets/loss_mask.
                    gm.data.Seq2SeqTask(
                        # Select which field from the dataset to use.
                        in_prompt=self.config["seq2seq_in_prompt"],
                        in_response=self.config["seq2seq_in_response"],
                        # Output batch is {'input': ..., 'target': ..., 'loss_mask': ...}
                        out_input="input",
                        out_target="target",
                        out_target_mask="loss_mask",
                        tokenizer=self.tokenizer,
                        # Padding parameters
                        max_length=self.config["seq2seq_max_length"],
                        truncate=self.config["seq2seq_truncate"],
                    ),
                ],
            )
            self.data = ds
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset: {str(e)}")


def create_pipeline(config: Dict) -> DataPipeline:
    """Create a data pipeline based on the configuration."""
    data_source = config.get("source", "").lower()
    if data_source == "huggingface":
        return HuggingFacePipeline(config)
    elif data_source == "tensorflow":
        return TensorFlowPipeline(config)
    elif data_source == "json":
        return JSONPipeline(config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")


def process_data(config: Dict) -> Dict:
    """Process data using the specified pipeline configuration."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
    pipeline = create_pipeline(config)
    pipeline.load_data()
    return pipeline.get_dataset()
