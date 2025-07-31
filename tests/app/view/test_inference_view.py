from unittest.mock import MagicMock

from streamlit.testing.v1 import AppTest

from tests.app.view.utils import setup_di_mock


def mock_inferencer():
    """Mock inferencer for testing."""
    inferencer = MagicMock()
    inferencer.list_checkpoints.return_value = [
        "checkpoint1",
        "checkpoint2",
        "checkpoint3",
    ]
    inferencer.is_loaded.return_value = True
    inferencer.load_model.return_value = True
    inferencer.delete_checkpoint.return_value = True
    inferencer.generate.return_value = "Generated response text"
    inferencer.count_tokens.return_value = 10
    return inferencer


def setup_inference_di_mock(monkeypatch, inferencer):
    """Helper to setup DI container mocking for inference."""

    def mock_get_service(name):
        if name == "inferencer":
            return inferencer
        return MagicMock()

    monkeypatch.setattr("services.di_container.get_service", mock_get_service)


def test_shows_inference_view_title_and_sections(monkeypatch):
    setup_inference_di_mock(monkeypatch, mock_inferencer())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()

    assert at.title[0].value == "Inference Playground"
    assert at.subheader[0].value == "Checkpoint Management"
    assert at.subheader[1].value == "Inference Playground"
