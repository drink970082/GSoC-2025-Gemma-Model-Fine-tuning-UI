from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

from tests.app.utils import setup_di_mock

from tests.app.components.inference.utils import mock_inferencer


def test_shows_checkpoint_selection_with_checkpoints(
    monkeypatch, mock_inferencer
):
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()
    assert (
        at.selectbox(key="checkpoint_selection").label
        == "Select a checkpoint for inference:"
    )
    assert any(btn.label == "Load checkpoint" for btn in at.button)
    assert any(btn.label == "Delete" for btn in at.button)


def test_load_checkpoint_success(monkeypatch, mock_inferencer):
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()

    at.button(key="load_checkpoint").click()
    at.run()

    mock_inferencer.load_model.assert_called_once_with("checkpoint1")
    assert "Loaded checkpoint: checkpoint1" in [
        success.value for success in at.success
    ]


def test_load_checkpoint_failure(monkeypatch, mock_inferencer):
    mock_inferencer.load_model.return_value = False
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()

    at.button(key="load_checkpoint").click()
    at.run()

    assert "Failed to load checkpoint: checkpoint1" in [
        error.value for error in at.error
    ]


def test_delete_checkpoint_success(monkeypatch, mock_inferencer):
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()

    at.button(key="delete_checkpoint").click()
    at.run()

    mock_inferencer.delete_checkpoint.assert_called_once_with("checkpoint1")
    assert "Deleted checkpoint: checkpoint1" in [
        success.value for success in at.success
    ]


def test_delete_checkpoint_failure(monkeypatch, mock_inferencer):
    mock_inferencer.delete_checkpoint.return_value = False
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()

    at.button(key="delete_checkpoint").click()
    at.run()

    assert "Failed to delete checkpoint." in [error.value for error in at.error]


def test_no_checkpoints_warning(monkeypatch, mock_inferencer):
    mock_inferencer.list_checkpoints.return_value = []
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()

    assert "No checkpoints found." in [warning.value for warning in at.warning]
