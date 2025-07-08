import os
import atexit
from itertools import islice
from typing import List, Dict, Tuple
from config.dataclass import DataConfig

import pandas as pd
import numpy as np
from gemma import gm
from kauldron import kd


class DataPipeline:
    """Base class for data pipelines."""

    def __init__(self, config: DataConfig):
        """Initialize the data pipeline with configuration."""
        self.config = config
        # The tokenizer is now instantiated directly here.
        self.tokenizer = gm.text.Gemma3Tokenizer()

    def _get_preview_split(self, num_records: int = 5) -> str:
        """Helper function to construct the preview split."""
        base_split = self.config.split or "train"
        split_type = base_split.split("[")[0]
        preview_split = f"{split_type}[:{num_records}]"
        return preview_split

    def _get_transforms(self) -> List[gm.data.Seq2SeqTask]:
        """Helper function to construct the list of transforms."""
        return [
            gm.data.Seq2SeqTask(
                in_prompt=self.config.seq2seq_in_prompt,
                in_response=self.config.seq2seq_in_response,
                max_length=self.config.seq2seq_max_length,
                truncate=self.config.seq2seq_truncate,
                out_input="input",
                out_target="target",
                out_target_mask="loss_mask",
                tokenizer=self.tokenizer,
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

    def _to_py_str(self, val):
        if isinstance(val, bytes):
            return val.decode("utf-8")
        if isinstance(val, (np.bytes_,)):
            return val.decode("utf-8")
        if isinstance(val, (str, np.str_)):
            return str(val)
        return val

    def get_prompt_and_response_df(
        self, data: Dict, prompt_key: str, response_key: str
    ) -> Tuple[List[str], List[str]]:
        if prompt_key not in data:
            raise ValueError(
                f"Source field name {prompt_key} not found in the dataset, possible options are: {list(data.keys())}"
            )
        if response_key not in data:
            raise ValueError(
                f"Target field name {response_key} not found in the dataset, possible options are: {list(data.keys())}"
            )
        src_texts = [self._to_py_str(ex) for ex in data[prompt_key]]
        dst_texts = [self._to_py_str(ex) for ex in data[response_key]]
        df = pd.DataFrame({"Prompt": src_texts, "Response": dst_texts})
        return df


class JSONPipeline(DataPipeline):
    """Pipeline for JSON format data with preview capabilities."""

    def __init__(self, config: DataConfig):
        super().__init__(config)
        self.temp_file_path = self.config.dataset_name
        if self.temp_file_path:
            atexit.register(self.cleanup)

    def get_raw_preview(self, num_records: int = 5) -> List[Dict]:
        """Efficiently reads the first N records from a JSONL file."""
        try:
            df = pd.read_json(
                self.temp_file_path,
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
                path=self.temp_file_path,
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
                path=self.temp_file_path,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size,
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

    def get_raw_preview(self, num_records: int = 5) -> pd.DataFrame:
        """Loads the first N records of the raw dataset WITHOUT transforms."""
        try:
            ds = kd.data.py.Tfds(
                name=self.config.dataset_name,
                split=self._get_preview_split(),
                shuffle=False,
                batch_size=self.config.batch_size,
            )
            return self.get_prompt_and_response_df(
                ds[0],
                self.config.seq2seq_in_prompt,
                self.config.seq2seq_in_response,
            )
        except Exception as e:
            raise ValueError(f"Failed to load raw TFDS preview: {str(e)}")

    def get_tokenized_preview(self, num_records: int = 5) -> List[Dict]:
        """Loads and processes the first N records WITH transforms."""
        try:
            ds = kd.data.py.Tfds(
                name=self.config.dataset_name,
                split=self._get_preview_split(),
                shuffle=False,
                batch_size=self.config.batch_size,
                transforms=self._get_transforms(),
            )
            return ds[0]
        except Exception as e:
            raise ValueError(f"Failed to load tokenized TFDS preview: {str(e)}")

    def get_train_dataset(self) -> kd.data.Pipeline:
        """Loads the full dataset for actual training."""
        try:
            return kd.data.py.Tfds(
                name=self.config.dataset_name,
                split=self.config.split,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size,
                transforms=self._get_transforms(),
            )
        except Exception as e:
            raise ValueError(f"Failed to load TensorFlow dataset: {str(e)}")


class HuggingFacePipeline(DataPipeline):
    """Pipeline for HuggingFace datasets with preview capabilities."""

    def get_raw_preview(self, num_records: int = 5) -> pd.DataFrame:
        """Streams the first N records of the raw dataset WITHOUT transforms."""
        try:
            ds = kd.data.py.HuggingFace(
                path=self.config.dataset_name,
                config=self.config.config,
                split=self._get_preview_split(),
                shuffle=False,
                batch_size=self.config.batch_size,
            )
            return self.get_prompt_and_response_df(
                ds[0],
                self.config.seq2seq_in_prompt,
                self.config.seq2seq_in_response,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load raw HuggingFace preview: {str(e)}"
            )

    def get_tokenized_preview(self, num_records: int = 5) -> List[Dict]:
        """Streams and processes the first N records WITH transforms."""
        try:
            ds = kd.data.py.HuggingFace(
                path=self.config.dataset_name,
                config=self.config.config,
                split=self._get_preview_split(),
                shuffle=False,
                batch_size=self.config.batch_size,
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
                path=self.config.dataset_name,
                config=self.config.config,
                split=self.config.split,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size,
                transforms=self._get_transforms(),
            )
        except Exception as e:
            raise ValueError(f"Failed to load HuggingFace dataset: {str(e)}")


def create_pipeline(config: DataConfig) -> DataPipeline:
    """Create a data pipeline based on the configuration."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
    data_source = config.source.lower()
    # The tokenizer is no longer passed as an argument here.
    if data_source == "huggingface":
        return HuggingFacePipeline(config)
    elif data_source == "tensorflow":
        return TensorFlowPipeline(config)
    elif data_source == "json":
        return JSONPipeline(config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
