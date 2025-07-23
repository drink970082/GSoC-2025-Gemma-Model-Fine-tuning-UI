from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def mock_training_service():
    svc = MagicMock()
    svc.is_training_running.return_value = "RUNNING"
    return svc

# Running Training
def test_shows_running_training(monkeypatch, mock_training_service):
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(mock_training_service)
    )
    at.run()
    assert "An active fine-tuning process is running." in at.info[0].value
    assert at.button[0].label == "Go to Live Monitoring"
    assert at.button[1].label == "Abort and Start New"


def test_running_training_go_to_live_monitoring(
    monkeypatch, mock_training_service
):
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(mock_training_service)
    )
    at.run()
    at.button[0].click()  # "Go to Live Monitoring"
    assert at.session_state["view"] == "training_dashboard"


def test_abort_confirmation(monkeypatch):
    svc = MagicMock()
    svc.is_training_running.return_value = "RUNNING"
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(svc)
    )
    at.session_state["abort_confirmation"] = True
    at.run()
    assert "Are you sure you want to abort" in at.warning[0].value
    assert at.button[0].label == "Yes, Abort"
    assert at.button[1].label == "No, Cancel"


def test_running_training_abort_and_start_new(
    monkeypatch, mock_training_service
):
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(mock_training_service)
    )
    at.run()
    at.button[1].click()  # "Abort and Start New"
    assert at.session_state["abort_confirmation"] is True


def test_abort_confirmation_yes_aborts(monkeypatch):
    svc = MagicMock()
    svc.is_training_running.return_value = "RUNNING"
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(svc)
    )
    at.session_state["abort_confirmation"] = True
    at.run()
    at.button[0].click()  # "Yes, Abort"
    assert at.session_state["abort_confirmation"] is False
    assert at.session_state["view"] == "create_model"
    svc.stop_training.assert_called_once_with(mode="force")


def test_abort_confirmation_no_cancels(monkeypatch):
    svc = MagicMock()
    svc.is_training_running.return_value = "RUNNING"
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(svc)
    )
    at.session_state["abort_confirmation"] = True
    at.run()
    at.button[1].click()  # "No, Cancel"
    assert at.session_state["abort_confirmation"] is False
    svc.stop_training.assert_not_called()


# Main Navigation
def test_shows_main_navigation(monkeypatch):
    svc = MagicMock()
    svc.is_training_running.return_value = "IDLE"
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(svc)
    )
    at.run()
    assert "Choose an option to get started." in at.info[0].value
    assert at.button[0].label == "Start New Fine-Tuning"
    assert at.button[1].label == "Inference Existing Model"


def test_main_navigation_start_new(monkeypatch):
    svc = MagicMock()
    svc.is_training_running.return_value = "IDLE"
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(svc)
    )
    at.run()
    at.button[0].click()  # "Start New Fine-Tuning"
    assert at.session_state["view"] == "create_model"


def test_main_navigation_inference(monkeypatch):
    svc = MagicMock()
    svc.is_training_running.return_value = "IDLE"
    at = AppTest.from_function(
        lambda: __import__(
            "app.view.welcome_view"
        ).view.welcome_view.show_welcome_modal(svc)
    )
    at.run()
    at.button[1].click()  # "Inference Existing Model"
    assert at.session_state["view"] == "inference"
