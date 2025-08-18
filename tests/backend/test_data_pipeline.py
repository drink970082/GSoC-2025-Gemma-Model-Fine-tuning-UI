import atexit
import os
from typing import Any, Dict
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from backend.data_pipeline import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_PREVIEW_RECORDS,
    DEFAULT_SPLIT,
    DataPipeline,
    HuggingFacePipeline,
    JSONPipeline,
    TensorFlowPipeline,
    create_pipeline,
)
from config.dataclass import DataConfig

# --- Test Fixtures ---


@pytest.fixture
def sample_config():
    return DataConfig(
        source="json",
        dataset_name="test_dataset",
        split="train",
        shuffle=True,
        batch_size=4,
        seq2seq_in_prompt="src",
        seq2seq_in_response="dst",
        seq2seq_max_length=200,
        seq2seq_truncate=True,
        config=None,
    )


@pytest.fixture
def sample_data():
    return {
        "src": ["Hello", "Hi", "Good morning"],
        "dst": ["Bonjour", "Salut", "Bonjour"],
        "other_field": ["ignore", "ignore", "ignore"],
    }


# --- Base DataPipeline Tests ---


def test_data_pipeline_init(sample_config):
    """Test DataPipeline initialization."""
    pipeline = DataPipeline(sample_config)
    assert pipeline.config == sample_config
    assert pipeline.tokenizer is not None


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_data_pipeline_get_preview_success(
    mock_tokenizer, sample_config, sample_data
):
    """Test successful preview generation."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    # Mock the pipeline to return sample data
    with patch.object(pipeline, "_get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.__getitem__.return_value = sample_data
        mock_get_pipeline.return_value = mock_pipeline

        result = pipeline.get_preview(tokenized=False)

        assert isinstance(result, pd.DataFrame)
        assert "Prompt" in result.columns
        assert "Response" in result.columns
        assert len(result) == 3
        assert result["Prompt"].iloc[0] == "Hello"
        assert result["Response"].iloc[0] == "Bonjour"


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_data_pipeline_get_preview_tokenized(mock_tokenizer, sample_config):
    """Test preview with tokenized=True."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    with patch.object(pipeline, "_get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_pipeline.__getitem__.return_value = {"tokenized": "data"}
        mock_get_pipeline.return_value = mock_pipeline

        result = pipeline.get_preview(tokenized=True)
        assert result == {"tokenized": "data"}


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_data_pipeline_get_preview_exception(mock_tokenizer, sample_config):
    """Test preview with exception handling."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    with patch.object(
        pipeline, "_get_pipeline", side_effect=Exception("Test error")
    ):
        with pytest.raises(ValueError, match="Failed to load json preview"):
            pipeline.get_preview()


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_data_pipeline_get_train_dataset_success(mock_tokenizer, sample_config):
    """Test successful train dataset loading."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    with patch.object(pipeline, "_get_pipeline") as mock_get_pipeline:
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline

        result = pipeline.get_train_dataset()
        assert result == mock_pipeline


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_data_pipeline_get_train_dataset_exception(
    mock_tokenizer, sample_config
):
    """Test train dataset loading with exception."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    with patch.object(
        pipeline, "_get_pipeline", side_effect=Exception("Test error")
    ):
        with pytest.raises(ValueError, match="Failed to load json data"):
            pipeline.get_train_dataset()


def test_data_pipeline_get_prompt_and_response_df_success(sample_config):
    """Test successful prompt/response DataFrame creation."""
    pipeline = DataPipeline(sample_config)
    data = {
        "src": ["Hello", "Hi"],
        "dst": ["Bonjour", "Salut"],
    }

    result = pipeline._get_prompt_and_response_df(data, "src", "dst")

    assert isinstance(result, pd.DataFrame)
    assert result["Prompt"].iloc[0] == "Hello"
    assert result["Response"].iloc[0] == "Bonjour"


def test_data_pipeline_get_prompt_and_response_df_missing_prompt(sample_config):
    """Test DataFrame creation with missing prompt field."""
    pipeline = DataPipeline(sample_config)
    data = {"dst": ["Bonjour"]}

    with pytest.raises(ValueError, match="Source field name src not found"):
        pipeline._get_prompt_and_response_df(data, "src", "dst")


def test_data_pipeline_get_prompt_and_response_df_missing_response(
    sample_config,
):
    """Test DataFrame creation with missing response field."""
    pipeline = DataPipeline(sample_config)
    data = {"src": ["Hello"]}

    with pytest.raises(ValueError, match="Target field name dst not found"):
        pipeline._get_prompt_and_response_df(data, "src", "dst")


def test_data_pipeline_get_preview_split_default(sample_config):
    """Test preview split with default split."""
    pipeline = DataPipeline(sample_config)
    result = pipeline._get_preview_split()
    assert result == f"train[:{DEFAULT_PREVIEW_RECORDS}]"


def test_data_pipeline_get_preview_split_custom(sample_config):
    """Test preview split with custom split."""
    sample_config.split = "test[0:100]"
    pipeline = DataPipeline(sample_config)
    result = pipeline._get_preview_split()
    assert result == f"test[:{DEFAULT_PREVIEW_RECORDS}]"


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
@patch("backend.data_pipeline.gm.data.Seq2SeqTask")
def test_data_pipeline_get_transforms(mock_tokenizer, sample_config):
    """Test transform creation."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    transforms = pipeline._get_transforms()

    assert len(transforms) == 1
    assert isinstance(transforms[0], MagicMock)  # gm.data.Seq2SeqTask


def test_data_pipeline_to_py_str_various_types(sample_config):
    """Test string conversion with various input types."""
    pipeline = DataPipeline(sample_config)

    assert pipeline._to_py_str("hello") == "hello"
    assert pipeline._to_py_str(b"hello") == "hello"
    assert pipeline._to_py_str(np.bytes_(b"hello")) == "hello"
    assert pipeline._to_py_str(np.str_("hello")) == "hello"
    assert pipeline._to_py_str(123) == "123"


def test_data_pipeline_get_pipeline_not_implemented(sample_config):
    """Test that base class raises NotImplementedError."""
    pipeline = DataPipeline(sample_config)

    with pytest.raises(NotImplementedError):
        pipeline._get_pipeline()


# --- JSONPipeline Tests ---


def test_json_pipeline_init(sample_config):
    """Test JSONPipeline initialization."""
    pipeline = JSONPipeline(sample_config)
    assert pipeline.temp_file_path == "test_dataset"


def test_json_pipeline_init_no_dataset_name():
    """Test JSONPipeline initialization without dataset name."""
    config = DataConfig(
        source="json",
        dataset_name=None,
        split="train",
        shuffle=True,
        batch_size=4,
        seq2seq_in_prompt="src",
        seq2seq_in_response="dst",
        seq2seq_max_length=200,
        seq2seq_truncate=True,
        config=None,
    )
    pipeline = JSONPipeline(config)
    assert pipeline.temp_file_path is None


@patch("backend.data_pipeline.kd.data.py.Json")
@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_json_pipeline_get_pipeline_tokenized(
    mock_tokenizer, mock_json, sample_config
):
    """Test JSONPipeline with tokenized=True."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = JSONPipeline(sample_config)

    result = pipeline._get_pipeline(tokenized=True, preview=False)

    mock_json.assert_called_once()
    call_args = mock_json.call_args
    assert call_args[1]["path"] == "test_dataset"
    assert call_args[1]["shuffle"] is True
    assert call_args[1]["batch_size"] == 4
    assert "transforms" in call_args[1]


@patch("backend.data_pipeline.kd.data.py.Json")
@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_json_pipeline_get_pipeline_not_tokenized(
    mock_tokenizer, mock_json, sample_config
):
    """Test JSONPipeline with tokenized=False."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = JSONPipeline(sample_config)

    result = pipeline._get_pipeline(tokenized=False, preview=False)

    mock_json.assert_called_once()
    call_args = mock_json.call_args
    assert call_args[1]["path"] == "test_dataset"
    assert call_args[1]["shuffle"] is True
    assert call_args[1]["batch_size"] == 4
    assert "transforms" not in call_args[1]


@patch("backend.data_pipeline.os.unlink")
@patch("backend.data_pipeline.os.path.exists", return_value=True)
def test_json_pipeline_cleanup(mock_exists, mock_unlink, sample_config):
    """Test JSONPipeline cleanup."""
    pipeline = JSONPipeline(sample_config)
    pipeline._cleanup()

    mock_exists.assert_called_once_with("test_dataset")
    mock_unlink.assert_called_once_with("test_dataset")


@patch("backend.data_pipeline.os.unlink")
@patch("backend.data_pipeline.os.path.exists", return_value=False)
def test_json_pipeline_cleanup_file_not_exists(
    mock_exists, mock_unlink, sample_config
):
    """Test JSONPipeline cleanup when file doesn't exist."""
    pipeline = JSONPipeline(sample_config)
    pipeline._cleanup()

    mock_exists.assert_called_once_with("test_dataset")
    mock_unlink.assert_not_called()


def test_json_pipeline_cleanup_no_temp_file(sample_config):
    """Test JSONPipeline cleanup with no temp file."""
    pipeline = JSONPipeline(sample_config)
    pipeline._cleanup()  # Should not raise any exception


# --- TensorFlowPipeline Tests ---


@patch("backend.data_pipeline.kd.data.py.Tfds")
@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_tensorflow_pipeline_get_pipeline_tokenized(
    mock_tokenizer, mock_tfds, sample_config
):
    """Test TensorFlowPipeline with tokenized=True."""
    sample_config.source = "tensorflow"
    mock_tokenizer.return_value = MagicMock()
    pipeline = TensorFlowPipeline(sample_config)

    result = pipeline._get_pipeline(tokenized=True, preview=False)

    mock_tfds.assert_called_once()
    call_args = mock_tfds.call_args
    assert call_args[1]["name"] == "test_dataset"
    assert call_args[1]["split"] == "train"
    assert call_args[1]["shuffle"] is True
    assert call_args[1]["batch_size"] == 4
    assert "transforms" in call_args[1]


@patch("backend.data_pipeline.kd.data.py.Tfds")
@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_tensorflow_pipeline_get_pipeline_preview(
    mock_tokenizer, mock_tfds, sample_config
):
    """Test TensorFlowPipeline with preview=True."""
    sample_config.source = "tensorflow"
    mock_tokenizer.return_value = MagicMock()
    pipeline = TensorFlowPipeline(sample_config)

    result = pipeline._get_pipeline(tokenized=False, preview=True)

    mock_tfds.assert_called_once()
    call_args = mock_tfds.call_args
    assert call_args[1]["split"] == f"train[:{DEFAULT_PREVIEW_RECORDS}]"


# --- HuggingFacePipeline Tests ---


@patch("backend.data_pipeline.kd.data.py.HuggingFace")
@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_huggingface_pipeline_get_pipeline_tokenized(
    mock_tokenizer, mock_hf, sample_config
):
    """Test HuggingFacePipeline with tokenized=True."""
    sample_config.source = "huggingface"
    sample_config.config = "test_config"
    mock_tokenizer.return_value = MagicMock()
    pipeline = HuggingFacePipeline(sample_config)

    result = pipeline._get_pipeline(tokenized=True, preview=False)

    mock_hf.assert_called_once()
    call_args = mock_hf.call_args
    assert call_args[1]["path"] == "test_dataset"
    assert call_args[1]["config"] == "test_config"
    assert call_args[1]["split"] == "train"
    assert call_args[1]["shuffle"] is True
    assert call_args[1]["batch_size"] == 4
    assert "transforms" in call_args[1]


@patch("backend.data_pipeline.kd.data.py.HuggingFace")
@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_huggingface_pipeline_get_pipeline_preview(
    mock_tokenizer, mock_hf, sample_config
):
    """Test HuggingFacePipeline with preview=True."""
    sample_config.source = "huggingface"
    mock_tokenizer.return_value = MagicMock()
    pipeline = HuggingFacePipeline(sample_config)

    result = pipeline._get_pipeline(tokenized=False, preview=True)

    mock_hf.assert_called_once()
    call_args = mock_hf.call_args
    assert call_args[1]["split"] == f"train[:{DEFAULT_PREVIEW_RECORDS}]"


# --- Factory Function Tests ---


@patch.dict("backend.data_pipeline.os.environ", {}, clear=True)
def test_create_pipeline_huggingface(sample_config):
    """Test create_pipeline with HuggingFace source."""
    sample_config.source = "huggingface"

    result = create_pipeline(sample_config)

    assert isinstance(result, HuggingFacePipeline)
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "1.00"


@patch.dict("backend.data_pipeline.os.environ", {}, clear=True)
def test_create_pipeline_tensorflow(sample_config):
    """Test create_pipeline with TensorFlow source."""
    sample_config.source = "tensorflow"

    result = create_pipeline(sample_config)

    assert isinstance(result, TensorFlowPipeline)
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "1.00"


@patch.dict("backend.data_pipeline.os.environ", {}, clear=True)
def test_create_pipeline_json(sample_config):
    """Test create_pipeline with JSON source."""
    sample_config.source = "json"

    result = create_pipeline(sample_config)

    assert isinstance(result, JSONPipeline)
    assert os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION") == "1.00"


@patch.dict("backend.data_pipeline.os.environ", {}, clear=True)
def test_create_pipeline_unknown_source(sample_config):
    """Test create_pipeline with unknown source."""
    sample_config.source = "unknown"

    with pytest.raises(ValueError, match="Unknown data source: unknown"):
        create_pipeline(sample_config)


# --- Edge Cases and Error Handling ---


def test_data_pipeline_get_preview_split_none_split(sample_config):
    """Test preview split with None split."""
    sample_config.split = None
    pipeline = DataPipeline(sample_config)
    result = pipeline._get_preview_split()
    assert result == f"{DEFAULT_SPLIT}[:{DEFAULT_PREVIEW_RECORDS}]"


def test_data_pipeline_get_preview_split_empty_split(sample_config):
    """Test preview split with empty split."""
    sample_config.split = ""
    pipeline = DataPipeline(sample_config)
    result = pipeline._get_preview_split()
    assert result == f"{DEFAULT_SPLIT}[:{DEFAULT_PREVIEW_RECORDS}]"


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_data_pipeline_get_preview_exception_during_pipeline(
    mock_tokenizer, sample_config
):
    """Test preview with exception during pipeline creation."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    with patch.object(
        pipeline, "_get_pipeline", side_effect=Exception("Pipeline error")
    ):
        with pytest.raises(ValueError, match="Failed to load json preview"):
            pipeline.get_preview()


@patch("backend.data_pipeline.gm.text.Gemma3Tokenizer")
def test_data_pipeline_get_train_dataset_exception_during_pipeline(
    mock_tokenizer, sample_config
):
    """Test train dataset with exception during pipeline creation."""
    mock_tokenizer.return_value = MagicMock()
    pipeline = DataPipeline(sample_config)

    with patch.object(
        pipeline, "_get_pipeline", side_effect=Exception("Pipeline error")
    ):
        with pytest.raises(ValueError, match="Failed to load json data"):
            pipeline.get_train_dataset()
