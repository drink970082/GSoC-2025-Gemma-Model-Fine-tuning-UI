from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock


# Poll Training Status Tests
def test_poll_training_status_running_shows_info(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    assert "Training in progress." in [info.value for info in at.info]


def test_poll_training_status_finished_triggers_rerun(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="FINISHED"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["session_started_by_app"] = True
    at.run()
    assert at.session_state["session_started_by_app"] is False


def test_poll_training_status_failed_triggers_rerun(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="FAILED"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["session_started_by_app"] = True
    at.run()
    assert at.session_state["session_started_by_app"] is False


def test_poll_training_status_abort_early_return(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.run()
    assert "Training in progress." not in [info.value for info in at.info]


def test_poll_training_status_idle_no_info(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="IDLE"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    assert "Training in progress." not in [info.value for info in at.info]


# Main View Tests
def test_show_training_dashboard_view_renders_title(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    assert at.title[0].value == "LLM Fine-Tuning Dashboard"


def test_show_training_dashboard_view_renders_sections(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    print([subheader.value for subheader in at.subheader])
    assert "Key Performance Indicators" in [
        subheader.value for subheader in at.subheader
    ]
    assert "Core Performance Plots" in [
        subheader.value for subheader in at.subheader
    ]
    assert "System Resource Usage" in [
        subheader.value for subheader in at.subheader
    ]


def test_show_training_dashboard_view_calls_panels(monkeypatch):
    svc = mock_training_service(status="RUNNING")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    svc.is_training_running.assert_called()


# Additional Session State Tests
def test_poll_training_status_session_started_false_no_rerun(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="FINISHED"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["session_started_by_app"] = False
    at.run()
    assert at.session_state["session_started_by_app"] is False


def test_poll_training_status_abort_during_polling(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = False
    at.run()
    at.session_state["abort_training"] = True
    at.run()
    assert "Training in progress." not in [info.value for info in at.info]


# Training Status Edge Cases
def test_poll_training_status_none_status(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status=None))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    assert "Training in progress." not in [info.value for info in at.info]


def test_poll_training_status_invalid_status(monkeypatch):
    setup_di_mock(monkeypatch, mock_training_service(status="INVALID_STATUS"))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    assert "Training in progress." not in [info.value for info in at.info]


def test_poll_training_status_service_exception(monkeypatch):
    svc = mock_training_service(status="RUNNING")
    svc.is_training_running.side_effect = Exception("Service error")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    assert "Training in progress." not in [info.value for info in at.info]


# Fragment Behavior Tests
def test_poll_training_status_transition_running_to_finished(
    monkeypatch,
):
    svc = mock_training_service(status="RUNNING")
    svc.is_training_running.side_effect = [
        "RUNNING",
        "RUNNING",
        "FINISHED",
        "FINISHED",
    ]
    setup_di_mock(monkeypatch, svc)
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["session_started_by_app"] = True
    at.run()
    assert "Training in progress." in [info.value for info in at.info]
    at.run()
    assert at.session_state["session_started_by_app"] is False


def test_poll_training_status_fragment_timing(monkeypatch):
    svc = mock_training_service(status="RUNNING")
    setup_di_mock(monkeypatch, svc)
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    initial_call_count = svc.is_training_running.call_count
    at.run()
    assert svc.is_training_running.call_count > initial_call_count


def test_poll_training_status_no_rerun_when_not_session_started(
    monkeypatch,
):
    setup_di_mock(monkeypatch, mock_training_service(status="FINISHED"))
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["session_started_by_app"] = False
    at.run()
    assert at.session_state["session_started_by_app"] is False
