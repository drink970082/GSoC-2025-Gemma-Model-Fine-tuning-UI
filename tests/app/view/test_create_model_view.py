from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock


@pytest.fixture
def mock_create_pipeline(monkeypatch):
    """Mock create_pipeline for pipeline creation.(avoid heavy imports)"""
    mock_pipeline = MagicMock()
    mock_pipeline.get_train_dataset.return_value = MagicMock()

    def mock_create_pipeline_func(config):
        return mock_pipeline

    monkeypatch.setattr(
        "app.components.create_model.start_training_button.create_pipeline",
        mock_create_pipeline_func,
    )
    return mock_pipeline


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


def test_start_training_success(monkeypatch, mock_create_pipeline):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "test_model"
    at.session_state["data_config"] = MagicMock()
    at.session_state["model_config"] = MagicMock()
    at.run()
    at.button(key="start_training_button").click().run()
    assert at.session_state["view"] == "training_dashboard"
    assert at.session_state["abort_training"] is False
    assert at.session_state["session_started_by_app"] is True


def test_start_training_failure(monkeypatch, mock_create_pipeline):
    svc = mock_training_service(status="FAILED")
    svc.start_training.return_value = False
    setup_di_mock(monkeypatch, svc)
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.session_state["model_name"] = "fail_model"
    at.session_state["data_config"] = MagicMock()
    at.session_state["model_config"] = MagicMock()
    at.run()
    at.button(key="start_training_button").click().run()
    print(at.error[0].value)
    assert any(
        "Training failed to start. Please try again." in err.value
        for err in at.error
    )
