import os
import atexit
from typing import Optional, Union, List
from config.dataclass import DataConfig

import pandas as pd
import numpy as np
from gemma import gm
from kauldron import kd

DEFAULT_PREVIEW_RECORDS = 5
DEFAULT_SPLIT = "train"
DEFAULT_BATCH_SIZE = 4


class DataPipeline:
    """Base class for data pipelines."""

    def __init__(self, config: DataConfig) -> None:
        """Initialize the data pipeline with configuration."""
        self.config: DataConfig = config
        self.tokenizer: gm.text.Gemma3Tokenizer = gm.text.Gemma3Tokenizer()

    def get_preview(self, tokenized: bool = False) -> pd.DataFrame:
        """Efficiently reads the first N records from a dataset."""
        try:
            ds = self._get_pipeline(tokenized=tokenized, preview=True)
            if tokenized:
                return pd.DataFrame(ds[0])
            else:
                return self._get_prompt_and_response_df(
                    ds[0],
                    self.config.seq2seq_in_prompt,
                    self.config.seq2seq_in_response,
                )
        except Exception as e:
            raise ValueError(
                f"Failed to load {self.config.source} preview: {str(e)}"
            )

    def get_train_dataset(self) -> kd.data.Pipeline:
        """Loads the full dataset from the JSONL file for training."""
        try:
            return self._get_pipeline(tokenized=True, preview=False)
        except Exception as e:
            raise ValueError(
                f"Failed to load {self.config.source} data: {str(e)}"
            )

    def _get_pipeline(
        self, tokenized: bool = False, preview: bool = False
    ) -> kd.data.Pipeline:
        """Helper function to get the dataset."""
        raise NotImplementedError

    def _get_prompt_and_response_df(
        self, data: dict, prompt_key: str, response_key: str
    ) -> pd.DataFrame:
        """Helper function to get the prompt and response dataframe."""
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
        return pd.DataFrame({"Prompt": src_texts, "Response": dst_texts})

    def _get_preview_split(self) -> str:
        """Helper function to construct the preview split."""
        base_split = self.config.split or DEFAULT_SPLIT
        split_type = base_split.split("[")[0]
        return f"{split_type}[:{DEFAULT_PREVIEW_RECORDS}]"

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

    def _to_py_str(self, val: Union[bytes, np.bytes_, str, np.str_]) -> str:
        """Helper function to convert various types to Python strings."""
        if isinstance(val, bytes):
            return val.decode("utf-8")
        if isinstance(val, (np.bytes_,)):
            return val.decode("utf-8")
        if isinstance(val, (str, np.str_)):
            return str(val)
        return str(val)


class JSONPipeline(DataPipeline):
    """Pipeline for JSON format data with preview capabilities."""

    def __init__(self, config: DataConfig) -> None:
        super().__init__(config)
        self.temp_file_path: Optional[str] = self.config.dataset_name
        if self.temp_file_path:
            atexit.register(self._cleanup)

    def _get_pipeline(
        self, tokenized: bool = False, preview: bool = False
    ) -> kd.data.Pipeline:
        """Helper function to get the dataset."""
        if tokenized:
            return kd.data.py.Json(
                path=self.temp_file_path,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size or DEFAULT_BATCH_SIZE,
                transforms=self._get_transforms(),
            )
        else:
            return kd.data.py.Json(
                path=self.temp_file_path,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size or DEFAULT_BATCH_SIZE,
            )

    def _cleanup(self) -> None:
        """Deletes the temporary file upon script termination."""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)


class TensorFlowPipeline(DataPipeline):
    """Pipeline for TensorFlow datasets with preview capabilities."""

    def _get_pipeline(
        self, tokenized: bool = False, preview: bool = False
    ) -> kd.data.Pipeline:
        """Helper function to get the dataset."""
        split = (
            self._get_preview_split()
            if preview
            else (self.config.split or DEFAULT_SPLIT)
        )
        if tokenized:
            return kd.data.py.Tfds(
                name=self.config.dataset_name,
                split=split,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size or DEFAULT_BATCH_SIZE,
                transforms=self._get_transforms(),
            )
        else:
            return kd.data.py.Tfds(
                name=self.config.dataset_name,
                split=split,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size or DEFAULT_BATCH_SIZE,
            )


class HuggingFacePipeline(DataPipeline):
    """Pipeline for HuggingFace datasets with preview capabilities."""

    def _get_pipeline(
        self, tokenized: bool = False, preview: bool = False
    ) -> kd.data.Pipeline:
        """Helper function to get the dataset."""
        split = (
            self._get_preview_split()
            if preview
            else (self.config.split or DEFAULT_SPLIT)
        )
        if tokenized:
            return kd.data.py.HuggingFace(
                path=self.config.dataset_name,
                config=self.config.config,
                split=split,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size or DEFAULT_BATCH_SIZE,
                transforms=self._get_transforms(),
            )
        else:
            return kd.data.py.HuggingFace(
                path=self.config.dataset_name,
                config=self.config.config,
                split=split,
                shuffle=self.config.shuffle,
                batch_size=self.config.batch_size or DEFAULT_BATCH_SIZE,
            )


def create_pipeline(config: DataConfig) -> DataPipeline:
    """Create a data pipeline based on the configuration."""
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
    data_source = config.source.lower()
    if data_source == "huggingface":
        return HuggingFacePipeline(config)
    elif data_source == "tensorflow":
        return TensorFlowPipeline(config)
    elif data_source == "json":
        return JSONPipeline(config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
