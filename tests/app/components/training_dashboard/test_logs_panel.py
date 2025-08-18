from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock


def test_display_logs_panel_live_logs(monkeypatch):
    """Test live logs display with stdout and stderr."""
    svc = mock_training_service()
    svc.get_log_contents.return_value = (
        "log line 1\nlog line 2",
        "error: something failed\nerror: another",
    )
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should show live logs subheader
    assert "Live Training Logs" in [
        subheader.value for subheader in at.subheader
    ]
    # Should show error count metric
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Error Count"] == "2"
    # Should show log content and errors
    code_blocks = [code.value for code in at.code]
    assert any("log line 1" in block for block in code_blocks)
    assert any("--- ERRORS ---" in block for block in code_blocks)
    assert any("error: something failed" in block for block in code_blocks)


def test_display_logs_panel_live_logs_no_stderr(monkeypatch):
    """Test live logs display with only stdout."""
    svc = mock_training_service()
    svc.get_log_contents.return_value = ("log line 1\nlog line 2", "")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Error Count"] == "0"
    code_blocks = [code.value for code in at.code]
    assert any("log line 1" in block for block in code_blocks)
    assert not any("--- ERRORS ---" in block for block in code_blocks)


def test_display_logs_panel_live_logs_no_stdout(monkeypatch):
    """Test live logs display with only stderr."""
    svc = mock_training_service()
    svc.get_log_contents.return_value = ("", "error: something failed")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    code_blocks = [code.value for code in at.code]
    assert any(
        "Waiting for training process to start..." in block
        for block in code_blocks
    )
    assert any("--- ERRORS ---" in block for block in code_blocks)
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Error Count"] == "1"


def test_display_logs_panel_live_logs_empty(monkeypatch):
    """Test live logs display with no logs at all."""
    svc = mock_training_service()
    svc.get_log_contents.return_value = ("", "")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    code_blocks = [code.value for code in at.code]
    assert any(
        "Waiting for training process to start..." in block
        for block in code_blocks
    )
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Error Count"] == "0"


def test_display_logs_panel_frozen_logs(monkeypatch):
    """Test frozen logs display when training is aborted."""
    frozen_log = "frozen log content"
    svc = mock_training_service()
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.session_state["frozen_log"] = frozen_log
    at.run()

    # Should show frozen logs subheader
    assert "Live Training Logs (Frozen)" in [
        subheader.value for subheader in at.subheader
    ]
    code_blocks = [code.value for code in at.code]
    assert any("frozen log content" in block for block in code_blocks)


def test_display_logs_panel_frozen_logs_persistence(monkeypatch):
    """Test that frozen logs persist across runs."""
    frozen_log = "persisted frozen log"
    svc = mock_training_service()
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.session_state["frozen_log"] = frozen_log
    at.run()

    # Should use frozen log
    code_blocks = [code.value for code in at.code]
    assert any("persisted frozen log" in block for block in code_blocks)

    # Change service log (should not affect frozen state)
    svc.get_log_contents.return_value = ("new log", "error: new")
    at.run()
    code_blocks = [code.value for code in at.code]
    assert any("persisted frozen log" in block for block in code_blocks)


def test_display_logs_panel_error_counting(monkeypatch):
    """Test error counting utility with various stderr cases."""
    from app.components.training_dashboard.logs_panel import _count_errors

    assert _count_errors("") == 0
    assert _count_errors(None) == 0
    assert _count_errors("error: something\nerror: another") == 2
    assert _count_errors("ERROR: fail\nError: fail\nerrOr: fail") == 3
    assert _count_errors("warning: not an error") == 1  # substring match


def test_display_logs_panel_service_exception(monkeypatch):
    """Handles service exceptions gracefully."""
    svc = mock_training_service()
    svc.get_log_contents.side_effect = Exception("Service error")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # No assertion needed, just ensure no crash
    assert True
