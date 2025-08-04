from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest
from tests.app.utils import get_default_config


@pytest.fixture
def mock_training_config():
    """Create a mock TrainingConfig for testing."""
    config = get_default_config()
    return config


def test_config_summary_with_valid_config(mock_training_config):
    """Test config summary displays JSON when config is provided."""
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test-model"
    at.session_state["data_config"] = mock_training_config.data_config
    at.session_state["model_config"] = mock_training_config.model_config
    at.run()
    assert at.json


def test_config_summary_without_config():
    """Test config summary shows warning when no config is provided."""
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = None
    at.session_state["data_config"] = None
    at.session_state["model_config"] = None
    at.run()
    assert "No configuration available to preview" in [
        warning.body for warning in at.warning
    ]


def test_config_summary_handles_error(mock_training_config):
    """Test config summary handles errors gracefully."""
    with patch(
        "app.components.create_model.config_summary.asdict"
    ) as mock_asdict:
        mock_asdict.side_effect = Exception("Test error")
        at = AppTest.from_file("app/main.py")
        at.session_state["view"] = "create_model"
        at.session_state["model_name"] = "test-model-error"
        at.session_state["data_config"] = mock_training_config.data_config
        at.session_state["model_config"] = mock_training_config.model_config
        at.run()
        assert "Error displaying configuration: Test error" in [
            error.body for error in at.error
        ]
