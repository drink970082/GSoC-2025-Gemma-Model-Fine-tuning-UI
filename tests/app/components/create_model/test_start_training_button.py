from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import (
    get_default_config,
    mock_create_pipeline,
    setup_di_mock,
)


def test_start_training_button_display(monkeypatch):
    """Test start training button is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    assert start_button is not None
    assert start_button.label == "Start Fine-tuning"


@patch("app.view.create_model_view._get_config")
def test_start_training_button_with_valid_config(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button with valid configuration."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config to return a valid training config
    mock_get_config.return_value = get_default_config()

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should not show any error messages
    error_messages = [error.body for error in at.error]
    assert len(error_messages) == 0


@patch("app.view.create_model_view._get_config")
def test_start_training_button_without_model_name(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - missing model name."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config to return None (invalid config)
    mock_get_config.return_value = None

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Please enter all the fields." in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_with_whitespace_model_name(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - whitespace only model name."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with whitespace model name
    config = get_default_config()
    config.model_name = "   "
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Please enter a model name" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_without_dataset_name_tensorflow(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - missing dataset name for TensorFlow."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with empty dataset name
    config = get_default_config()
    config.data_config.dataset_name = ""
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Please provide dataset name" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_without_dataset_name_json(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - missing dataset name for JSON upload."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with JSON source and empty dataset name
    config = get_default_config()
    config.data_config.source = "json"
    config.data_config.dataset_name = ""
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message for JSON upload
    error_messages = [error.body for error in at.error]
    assert "Please upload a JSON file" in error_messages


@patch("app.components.create_model.start_training_button.create_pipeline")
@patch("app.view.create_model_view._get_config")
def test_start_training_button_pipeline_creation_error(
    mock_get_config, mock_create_pipeline, monkeypatch
):
    """Test start training button validation - pipeline creation error."""
    setup_di_mock(monkeypatch, MagicMock())

    mock_get_config.return_value = get_default_config()
    mock_create_pipeline.side_effect = Exception("Test pipeline error")

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Error creating pipeline: Test pipeline error" in error_messages


@patch("app.components.create_model.start_training_button.create_pipeline")
@patch("app.view.create_model_view._get_config")
def test_start_training_button_pipeline_get_dataset_error(
    mock_get_config, mock_create_pipeline, monkeypatch
):
    """Test start training button validation - pipeline get_train_dataset error."""
    setup_di_mock(monkeypatch, MagicMock())

    mock_get_config.return_value = get_default_config()
    mock_pipeline = MagicMock()
    mock_pipeline.get_train_dataset.side_effect = Exception(
        "Test dataset error"
    )
    mock_create_pipeline.return_value = mock_pipeline
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should show error message
    error_messages = [error.body for error in at.error]
    assert "Error creating pipeline: Test dataset error" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_successful_validation(
    mock_get_config, mock_create_pipeline, monkeypatch
):
    """Test start training button successful validation."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config to return a valid training config
    mock_get_config.return_value = get_default_config()

    # Mock successful pipeline creation
    mock_pipeline = MagicMock()
    mock_pipeline.get_train_dataset.return_value = None
    mock_create_pipeline.return_value = mock_pipeline

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    # Should not show any error messages
    error_messages = [error.body for error in at.error]
    assert len(error_messages) == 0


@patch("app.view.create_model_view._get_config")
def test_start_training_button_invalid_batch_size(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - invalid batch size."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with invalid batch size
    config = get_default_config()
    config.data_config.batch_size = 0
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Batch size must be greater than 0" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_invalid_max_length(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - invalid max sequence length."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with invalid max length
    config = get_default_config()
    config.data_config.seq2seq_max_length = 0
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Maximum sequence length must be greater than 0" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_empty_prompt_field(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - empty prompt field."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with empty prompt field
    config = get_default_config()
    config.data_config.seq2seq_in_prompt = ""
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a prompt field name" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_empty_response_field(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - empty response field."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with empty response field
    config = get_default_config()
    config.data_config.seq2seq_in_response = ""
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a response field name" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_invalid_epochs(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - invalid epochs."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with invalid epochs
    config = get_default_config()
    config.model_config.epochs = 0
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Number of epochs must be greater than 0" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_invalid_learning_rate(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - invalid learning rate."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with invalid learning rate
    config = get_default_config()
    config.model_config.learning_rate = 0
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Learning rate must be greater than 0" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_invalid_lora_rank(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - invalid LoRA rank."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with LoRA method and invalid rank
    config = get_default_config()
    config.model_config.method = "LoRA"
    # Create LoRA parameters since they're None by default
    from config.dataclass import LoraParams

    config.model_config.parameters = LoraParams(lora_rank=0)
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "LoRA rank must be greater than 0" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_whitespace_prompt_field(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - whitespace only prompt field."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with whitespace prompt field
    config = get_default_config()
    config.data_config.seq2seq_in_prompt = "   "
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a prompt field name" in error_messages


@patch("app.view.create_model_view._get_config")
def test_start_training_button_whitespace_response_field(
    mock_get_config, monkeypatch, mock_create_pipeline
):
    """Test start training button validation - whitespace only response field."""
    setup_di_mock(monkeypatch, MagicMock())

    # Mock the config with whitespace response field
    config = get_default_config()
    config.data_config.seq2seq_in_response = "   "
    mock_get_config.return_value = config

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    start_button = at.button(key="start_training_button")
    start_button.click()
    at.run()

    error_messages = [error.body for error in at.error]
    assert "Please provide a response field name" in error_messages
