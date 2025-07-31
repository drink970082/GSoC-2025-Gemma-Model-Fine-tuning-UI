from streamlit.testing.v1 import AppTest

from tests.app.view.utils import (
    setup_di_mock,
    mock_training_service,
)


# Running Training Tests
def test_shows_running_training(monkeypatch):
    svc = mock_training_service(status="RUNNING")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.run()

    assert "An active fine-tuning process is running." in [
        info.value for info in at.info
    ]
    assert any(btn.label == "Go to Live Monitoring" for btn in at.button)
    assert any(btn.label == "Abort and Start New" for btn in at.button)


def test_running_training_go_to_live_monitoring(monkeypatch):
    svc = mock_training_service(status="RUNNING")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.run()

    at.button(key="go_to_live_monitoring").click()
    at.run()
    assert at.session_state["view"] == "training_dashboard"


def test_abort_and_start_new_shows_confirmation(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.run()

    at.button(key="abort_and_start_new").click()
    at.run()
    assert at.session_state["abort_confirmation"] is True


def test_abort_confirmation_dialog(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.session_state["abort_confirmation"] = True
    at.run()

    assert "Are you sure you want to abort the current training process?" in [
        warning.value for warning in at.warning
    ]
    assert any(btn.label == "Yes, Abort" for btn in at.button)
    assert any(btn.label == "No, Cancel" for btn in at.button)


def test_abort_confirmation_yes_aborts(monkeypatch):
    svc = mock_training_service(status="RUNNING")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.session_state["abort_confirmation"] = True
    at.run()

    at.button(key="yes_abort").click()
    at.run()

    assert at.session_state["abort_confirmation"] is False
    assert at.session_state["view"] == "create_model"
    svc.stop_training.assert_called_once_with(mode="force")


def test_abort_confirmation_no_cancels(monkeypatch):
    svc = mock_training_service(status="RUNNING")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.session_state["abort_confirmation"] = True
    at.run()

    at.button(key="no_cancel").click()
    at.run()

    assert at.session_state["abort_confirmation"] is False
    svc.stop_training.assert_not_called()


# Main Navigation Tests (when training is idle)
def test_shows_main_navigation(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="IDLE"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.run()

    assert "Choose an option to get started." in [
        info.value for info in at.info
    ]
    assert any(btn.label == "Start New Fine-Tuning" for btn in at.button)
    assert any(btn.label == "Inference Existing Model" for btn in at.button)


def test_main_navigation_start_new(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="IDLE"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.run()

    at.button(key="start_new_fine_tuning").click()
    at.run()
    assert at.session_state["view"] == "create_model"


def test_main_navigation_inference(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="IDLE"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "welcome"
    at.run()

    at.button(key="inference_existing_model").click()
    at.run()
    assert at.session_state["view"] == "inference"
