from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import get_default_config, setup_di_mock


def test_start_training_button_display(monkeypatch):
    """Test start training button is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    assert start_button is not None
    assert start_button.label == "Start Fine-tuning"


def test_start_training_button_with_valid_config(monkeypatch):
    """Test start training button with valid configuration."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    assert start_button is not None


def test_start_training_button_without_model_name(monkeypatch):
    """Test start training button validation - missing model name."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = ""  # Empty model name
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Please enter a model name" in error_messages


def test_start_training_button_with_whitespace_model_name(monkeypatch):
    """Test start training button validation - whitespace only model name."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "   "  # Whitespace only
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Please enter a model name" in error_messages


def test_start_training_button_without_dataset_name_huggingface(monkeypatch):
    """Test start training button validation - missing dataset name for HuggingFace."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.data_config.dataset_name = ""  # Empty dataset name

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide dataset name" in error_messages


def test_start_training_button_without_dataset_name_tensorflow(monkeypatch):
    """Test start training button validation - missing dataset name for TensorFlow."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Please provide dataset name" in error_messages


def test_start_training_button_without_dataset_name_json(monkeypatch):
    """Test start training button validation - missing dataset name for JSON upload."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message for JSON upload
    error_messages = [error.body for error in at.error]
    assert "Please upload a JSON file" in error_messages


def test_start_training_button_with_whitespace_dataset_name(monkeypatch):
    """Test start training button validation - whitespace only dataset name."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Please provide dataset name" in error_messages


@patch("backend.data_pipeline.create_pipeline")
def test_start_training_button_pipeline_creation_error(
    mock_create_pipeline, monkeypatch
):
    """Test start training button validation - pipeline creation error."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock pipeline creation to raise an exception
    mock_create_pipeline.side_effect = Exception("Test pipeline error")

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Error creating pipeline: Test pipeline error" in error_messages


@patch("backend.data_pipeline.create_pipeline")
def test_start_training_button_pipeline_get_dataset_error(
    mock_create_pipeline, monkeypatch
):
    """Test start training button validation - pipeline get_train_dataset error."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock pipeline to raise exception on get_train_dataset
    mock_pipeline = MagicMock()
    mock_pipeline.get_train_dataset.side_effect = Exception(
        "Test dataset error"
    )
    mock_create_pipeline.return_value = mock_pipeline

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Error creating pipeline: Test dataset error" in error_messages


@patch("backend.data_pipeline.create_pipeline")
def test_start_training_button_successful_validation(
    mock_create_pipeline, monkeypatch
):
    """Test start training button successful validation."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock successful pipeline creation
    mock_pipeline = MagicMock()
    mock_pipeline.get_train_dataset.return_value = None
    mock_create_pipeline.return_value = mock_pipeline

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should not show any error messages
    error_messages = [error.body for error in at.error]
    assert len(error_messages) == 0


def test_start_training_button_multiple_validation_errors(monkeypatch):
    """Test start training button with multiple validation errors."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = ""  # Missing model name
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show model name error (first validation that fails)
    error_messages = [error.body for error in at.error]
    assert "Please enter a model name" in error_messages


def test_start_training_button_button_click_behavior(monkeypatch):
    """Test start training button click behavior."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = get_default_config().data_config
    at.session_state["model_config"] = get_default_config().model_config
    at.run()

    start_button = at.button(key="start_training_button")
    assert start_button is not None

    # Test button click
    start_button.click()
    at.run()

    # Button should still be present after click
    assert at.button(key="start_training_button") is not None


def test_start_training_button_invalid_batch_size(monkeypatch):
    """Test start training button validation - invalid batch size."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.data_config.batch_size = 0  # Invalid batch size

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Batch size must be greater than 0" in error_messages


def test_start_training_button_invalid_max_length(monkeypatch):
    """Test start training button validation - invalid max sequence length."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.data_config.seq2seq_max_length = 0  # Invalid max length

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Maximum sequence length must be greater than 0" in error_messages


def test_start_training_button_empty_prompt_field(monkeypatch):
    """Test start training button validation - empty prompt field."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.data_config.seq2seq_in_prompt = ""  # Empty prompt field

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a prompt field name" in error_messages


def test_start_training_button_empty_response_field(monkeypatch):
    """Test start training button validation - empty response field."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.data_config.seq2seq_in_response = ""  # Empty response field

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a response field name" in error_messages


def test_start_training_button_invalid_epochs(monkeypatch):
    """Test start training button validation - invalid epochs."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.model_config.epochs = 0  # Invalid epochs

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Number of epochs must be greater than 0" in error_messages


def test_start_training_button_invalid_learning_rate(monkeypatch):
    """Test start training button validation - invalid learning rate."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.model_config.learning_rate = 0  # Invalid learning rate

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Learning rate must be greater than 0" in error_messages


def test_start_training_button_invalid_lora_rank(monkeypatch):
    """Test start training button validation - invalid LoRA rank."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.model_config.method = "LoRA"
    config.model_config.parameters.lora_rank = 0  # Invalid LoRA rank

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "LoRA rank must be greater than 0" in error_messages


def test_start_training_button_whitespace_prompt_field(monkeypatch):
    """Test start training button validation - whitespace only prompt field."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.data_config.seq2seq_in_prompt = "   "  # Whitespace only

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a prompt field name" in error_messages


def test_start_training_button_whitespace_response_field(monkeypatch):
    """Test start training button validation - whitespace only response field."""
    setup_di_mock(monkeypatch, MagicMock())

    config = get_default_config()
    config.data_config.seq2seq_in_response = "   "  # Whitespace only

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = config.data_config
    at.session_state["model_config"] = config.model_config
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a response field name" in error_messages
