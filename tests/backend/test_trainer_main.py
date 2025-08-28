import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from backend.trainer_main import main, parse_config
from config.dataclass import DataConfig, LoraParams, ModelConfig, TrainingConfig

# --- Test Fixtures ---


def create_test_config(method="Standard", parameters=None):
    """Helper to create test configs without repetition."""
    return {
        "model_name": "test_model",
        "model_config": {
            "model_variant": "gemma-2b",
            "epochs": 3,
            "learning_rate": 0.001,
            "method": method,
            **({"parameters": parameters} if parameters else {}),
        },
        "data_config": {
            "source": "json",
            "dataset_name": "test_dataset",
            "split": "train",
            "shuffle": True,
            "batch_size": 32,
            "seq2seq_in_prompt": "prompt",
            "seq2seq_in_response": "response",
            "seq2seq_max_length": 512,
            "seq2seq_truncate": True,
            "config": None,
        },
    }


@pytest.fixture
def sample_config():
    return create_test_config()


@pytest.fixture
def lora_config():
    return create_test_config("LoRA", {"lora_rank": 8})


# --- parse_config Tests ---


def test_parse_config_standard(sample_config):
    """Test parsing standard training config."""
    result = parse_config(sample_config)

    assert isinstance(result, TrainingConfig)
    assert result.model_name == "test_model"
    assert result.model_config.model_variant == "gemma-2b"
    assert result.model_config.epochs == 3
    assert result.model_config.learning_rate == 0.001
    assert result.model_config.method == "Standard"
    assert result.model_config.parameters is None
    assert isinstance(result.data_config, DataConfig)


def test_parse_config_lora(lora_config):
    """Test parsing LoRA training config."""
    result = parse_config(lora_config)

    assert isinstance(result, TrainingConfig)
    assert result.model_config.method == "LoRA"
    assert isinstance(result.model_config.parameters, LoraParams)
    assert result.model_config.parameters.lora_rank == 8


def test_parse_config_no_parameters():
    """Test parsing config without parameters field."""
    config = create_test_config()

    result = parse_config(config)
    assert result.model_config.parameters is None


# --- Main Function Tests ---


@patch("backend.trainer_main.create_parser")
@patch("backend.trainer_main.ModelTrainer")
def test_main_success(mock_trainer_class, mock_create_parser, sample_config):
    """Test successful main execution."""
    # Mock parser and args
    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.config = sample_config
    mock_args.work_dir = "/tmp/work"
    mock_create_parser.return_value = mock_parser
    mock_parser.parse_args.return_value = mock_args

    # Mock trainer
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer

    # Mock sys.exit to prevent actual exit
    with patch("sys.exit") as mock_exit:
        main()

    # Verify trainer was created and called
    mock_trainer_class.assert_called_once()
    mock_trainer.train.assert_called_once()

    # Verify exit was called with success code
    mock_exit.assert_called_with(0)


@patch("backend.trainer_main.create_parser")
def test_main_config_error(mock_create_parser, sample_config):
    """Test main with configuration error."""
    # Mock parser and args
    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.config = sample_config
    mock_args.work_dir = "/tmp/work"
    mock_create_parser.return_value = mock_parser
    mock_parser.parse_args.return_value = mock_args

    # Mock ModelTrainer to raise ValueError
    with patch(
        "backend.trainer_main.ModelTrainer",
        side_effect=ValueError("Config error"),
    ):
        with patch("sys.exit") as mock_exit:
            main()

    # Verify error handling
    mock_exit.assert_called_with(1)


@patch("backend.trainer_main.create_parser")
def test_main_unhandled_exception(mock_create_parser, sample_config):
    """Test main with unhandled exception."""
    # Mock parser and args
    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.config = sample_config
    mock_args.work_dir = "/tmp/work"
    mock_create_parser.return_value = mock_parser
    mock_parser.parse_args.return_value = mock_args

    # Mock ModelTrainer to raise unexpected exception
    with patch(
        "backend.trainer_main.ModelTrainer",
        side_effect=RuntimeError("Unexpected error"),
    ):
        with patch("sys.exit") as mock_exit:
            main()

    # Verify error handling
    mock_exit.assert_called_with(1)


@patch("backend.trainer_main.create_parser")
def test_main_successful_exit_code(mock_create_parser, sample_config):
    """Test main sets correct exit code on success."""
    # Mock parser and args
    mock_parser = MagicMock()
    mock_args = MagicMock()
    mock_args.config = sample_config
    mock_args.work_dir = "/tmp/work"
    mock_create_parser.return_value = mock_parser
    mock_parser.parse_args.return_value = mock_args

    # Mock trainer
    with patch("backend.trainer_main.ModelTrainer") as mock_trainer_class:
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer

        with patch("sys.exit") as mock_exit:
            main()

    # Verify successful exit
    mock_exit.assert_called_with(0)


# --- Edge Cases ---


def test_parse_config_empty_parameters():
    """Test parsing config with empty parameters dict."""
    config = create_test_config("LoRA", {})

    result = parse_config(config)
    # Empty dict is falsy, so no LoraParams should be created
    assert result.model_config.parameters is None


def test_parse_config_unknown_method():
    """Test parsing config with unknown method (should still work)."""
    config = create_test_config("UnknownMethod")

    result = parse_config(config)
    assert result.model_config.method == "UnknownMethod"
    assert result.model_config.parameters is None


def test_parse_config_lora_without_parameters():
    """Test LoRA method without parameters field."""
    config = create_test_config("LoRA")  # No parameters

    result = parse_config(config)
    assert result.model_config.method == "LoRA"
    assert result.model_config.parameters is None


def test_parse_config_lora_with_none_parameters():
    """Test LoRA method with None parameters."""
    config = create_test_config("LoRA")
    config["model_config"]["parameters"] = None

    result = parse_config(config)
    assert result.model_config.method == "LoRA"
    assert result.model_config.parameters is None
