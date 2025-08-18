from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest
from tests.app.utils import get_default_config
from unittest.mock import patch


@pytest.fixture
def mock_training_config():
    """Create a mock TrainingConfig for testing."""
    config = get_default_config()
    return config

@patch("app.view.create_model_view._get_config")
def test_config_summary_with_valid_config(mock_get_config, mock_training_config):
    mock_get_config.return_value = mock_training_config
    """Test config summary displays JSON when config is provided."""
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    assert at.json


@patch("app.view.create_model_view._get_config")
def test_config_summary_without_config(mock_get_config):
    mock_get_config.return_value = None
    """Test config summary shows warning when no config is provided."""
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    assert "No configuration available to preview" in [
        warning.body for warning in at.warning
    ]


@patch("app.view.create_model_view._get_config")
def test_config_summary_handles_error(mock_get_config, mock_training_config):
    """Test config summary handles errors gracefully."""
    with patch(
        "app.components.create_model.config_summary.asdict"
    ) as mock_asdict:
        mock_asdict.side_effect = Exception("Test error")
        at = AppTest.from_file("app/main.py")
        at.session_state["view"] = "create_model"
        at.run()
        assert "Error displaying configuration: Test error" in [
            error.body for error in at.error
        ]
