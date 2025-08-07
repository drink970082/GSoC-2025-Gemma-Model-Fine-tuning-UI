from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock, mock_create_pipeline, get_default_config



def test_renders_all_sections(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    # Check for all step headers
    assert "1. Model Name" in [subheader.value for subheader in at.subheader]
    assert "2. Data Source" in [subheader.value for subheader in at.subheader]
    assert "3. Model Selection" in [
        subheader.value for subheader in at.subheader
    ]
    assert "4. Configuration Preview" in [
        subheader.value for subheader in at.subheader
    ]
    assert "5. Start Training" in [
        subheader.value for subheader in at.subheader
    ]

@patch("app.view.create_model_view._get_config")
def test_start_training_success(mock_get_config, monkeypatch, mock_create_pipeline):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))
    mock_get_config.return_value = get_default_config()
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    at.button(key="start_training_button").click().run()
    assert at.session_state["view"] == "training_dashboard"
    assert at.session_state["abort_training"] is False
    assert at.session_state["session_started_by_app"] is True


@patch("app.view.create_model_view._get_config")
def test_start_training_failure(mock_get_config, monkeypatch, mock_create_pipeline):
    setup_di_mock(monkeypatch, mock_training_service(status="FAILED"))
    mock_get_config.return_value = get_default_config()
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    at.button(key="start_training_button").click().run()
    print(at.error[0].value)
    assert any(
        "Training Failed" in err.value
        for err in at.error
    )


