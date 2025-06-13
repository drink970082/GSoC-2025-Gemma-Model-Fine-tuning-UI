import os
import atexit
from itertools import islice
from typing import Dict, List

import pandas as pd
from gemma import gm
from kauldron import kd


class DataPipeline:
    """Base class for data pipelines."""

    def __init__(self, config: Dict):
        """Initialize the data pipeline with configuration."""
        self.config = config
        # The tokenizer is now instantiated directly here.
        self.tokenizer = gm.text.Gemma3Tokenizer()

    def _get_preview_split(self, num_records: int = 5) -> str:
        """Helper function to construct the preview split."""
        base_split = self.config.get("split") or "train"
        split_type = base_split.split("[")[0]
        preview_split = f"{split_type}[:{num_records}]"
        return preview_split

    def _get_transforms(self) -> List[gm.data.Seq2SeqTask]:
        """Helper function to construct the list of transforms."""
        return [
            gm.data.Seq2SeqTask(
                in_prompt=self.config["seq2seq_in_prompt"],
                in_response=self.config["seq2seq_in_response"],
                out_input="input",
                out_target="target",
                out_target_mask="loss_mask",
                tokenizer=self.tokenizer,
                max_length=self.config["seq2seq_max_length"],
                truncate=self.config.get("seq2seq_truncate", True),
            ),
        ]

    def get_raw_preview(self, num_records: int = 5) -> List[Dict]:
        """Loads the first N records of the raw dataset WITHOUT transforms."""
        raise NotImplementedError

    def get_tokenized_preview(self, num_records: int = 5) -> List[Dict]:
        """Loads and processes the first N records WITH transforms."""
        raise NotImplementedError

    def get_train_dataset(self) -> kd.data.Pipeline:
        """Loads the full dataset for actual training."""
        raise NotImplementedError


class JSONPipeline(DataPipeline):
    """Pipeline for JSON format data with preview capabilities."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.temp_file_path = self.config.get("data_path")
        if self.temp_file_path:
            atexit.register(self.cleanup)

    def get_raw_preview(self, num_records: int = 5) -> List[Dict]:
        """Efficiently reads the first N records from a JSONL file."""
        try:
            df = pd.read_json(
                self.config["data_path"],
                lines=True,
                nrows=num_records,
            )
            return df.to_dict(orient="records")
        except Exception as e:
            raise ValueError(f"Failed to load raw JSON preview: {str(e)}")

    def get_tokenized_preview(self, num_records: int = 5) -> List[Dict]:
        """Loads and processes the first N records from a JSONL file."""
        try:
            ds = kd.data.py.Json(
                path=self.config["data_path"],
                shuffle=False,
                transforms=self._get_transforms(),
            )
            return list(islice(ds, num_records))
        except Exception as e:
            raise ValueError(f"Failed to load tokenized JSON preview: {str(e)}")

    def get_train_dataset(self) -> kd.data.Pipeline:
        """Loads the full dataset from the JSONL file for training."""
        try:
            return kd.data.py.Json(
                path=self.config["data_path"],
                shuffle=self.config.get("shuffle", True),
                batch_size=self.config.get("batch_size", 4),
                transforms=self._get_transforms(),
            )
        except Exception as e:
            raise ValueError(f"Failed to load JSON data: {str(e)}")

    def cleanup(self):
        """Deletes the temporary file upon script termination."""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)


class TensorFlowPipeline(DataPipeline):
    """Pipeline for TensorFlow datasets with preview capabilities."""

    def get_raw_preview(self, num_records: int = 5) -> List[Dict]:
        """Loads the first N records of the raw dataset WITHOUT transforms."""
        try:
            ds = kd.data.py.Tfds(
                name=self.config["dataset_name"],
                split=self._get_preview_split(),
                shuffle=False,
                batch_size=self.config.get("batch_size", 8),
            )
            return ds[0]
        except Exception as e:
            raise ValueError(f"Failed to load raw TFDS preview: {str(e)}")

    def get_tokenized_preview(self, num_records: int = 5) -> List[Dict]:
        """Loads and processes the first N records WITH transforms."""
        try:
            ds = kd.data.py.Tfds(
                name=self.config["dataset_name"],
                split=self._get_preview_split(),
                shuffle=False,
                batch_size=self.config.get("batch_size", 8),
                transforms=self._get_transforms(),
            )
            return ds[0]
        except Exception as e:
            raise ValueError(f"Failed to load tokenized TFDS preview: {str(e)}")

    def get_train_dataset(self) -> kd.data.Pipeline:
        """Loads the full dataset for actual training."""
        try:
            return kd.data.py.Tfds(
                name=self.config["dataset_name"],
                split=self.config.get("split") or "train",
                shuffle=self.config.get("shuffle", True),
                batch_size=self.config.get("batch_size", 8),
                transforms=self._get_transforms(),
            )
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow dataset: {str(e)}")


class HuggingFacePipeline(DataPipeline):
    """Pipeline for HuggingFace datasets with preview capabilities."""

    def get_raw_preview(self, num_records: int = 5) -> List[Dict]:
        """Streams the first N records of the raw dataset WITHOUT transforms."""
        try:
            ds = kd.data.py.HuggingFace(
                path=self.config["dataset_name"],
                split=self._get_preview_split(),
                shuffle=False,
                batch_size=self.config.get("batch_size", 8),
            )
            return ds[0]
        except Exception as e:
            raise ValueError(
                f"Failed to load raw HuggingFace preview: {str(e)}"
            )

    def get_tokenized_preview(self, num_records: int = 5) -> List[Dict]:
        """Streams and processes the first N records WITH transforms."""
        try:
            ds = kd.data.py.HuggingFace(
                path=self.config["dataset_name"],
                split=self._get_preview_split(),
                streaming=True,
                shuffle=False,
                batch_size=self.config.get("batch_size", 8),
                transforms=self._get_transforms(),
            )
            return ds[0]
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenized HuggingFace preview: {str(e)}"
            )

    def get_train_dataset(self) -> kd.data.Pipeline:
        """Loads the full dataset from HuggingFace for training."""
        try:
            return kd.data.py.HuggingFace(
                path=self.config["dataset_name"],
                split=self.config.get("split") or "train",
                shuffle=self.config.get("shuffle", True),
                batch_size=self.config.get("batch_size", 8),
                transforms=self._get_transforms(),
            )
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset: {str(e)}")


def create_pipeline(config: Dict) -> DataPipeline:
    """Create a data pipeline based on the configuration."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
    data_source = config.get("source", "").lower()
    # The tokenizer is no longer passed as an argument here.
    if data_source == "huggingface":
        return HuggingFacePipeline(config)
    elif data_source == "tensorflow":
        return TensorFlowPipeline(config)
    elif data_source == "json":
        return JSONPipeline(config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
