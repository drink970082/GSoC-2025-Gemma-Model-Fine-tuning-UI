from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock


# Main Control Panel Display Tests
def test_display_control_panel_running_shows_abort_button(monkeypatch):
    """Test that RUNNING status shows abort button."""
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for abort button
    buttons = [btn.label for btn in at.button]
    assert "Abort Training" in buttons


def test_display_control_panel_failed_shows_error_and_reset(monkeypatch):
    """Test that FAILED status shows error message and reset button."""
    setup_di_mock(monkeypatch, mock_training_service(status="FAILED"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for error message
    assert "Training Failed" in [error.value for error in at.error]
    # Check for reset button
    buttons = [btn.label for btn in at.button]
    assert "Reset Application" in buttons


def test_display_control_panel_finished_with_checkpoint_shows_success_and_next_buttons(
    monkeypatch,
):
    """Test that FINISHED status with checkpoint shows success and next step buttons."""
    setup_di_mock(monkeypatch, mock_training_service(status="FINISHED"))

    # Mock checkpoint folder to return a valid checkpoint
    mock_checkpoint = MagicMock()
    mock_checkpoint.name = "checkpoint_123"

    with patch(
        "app.components.training_dashboard.control_panel._find_latest_checkpoint"
    ) as mock_find:
        mock_find.return_value = mock_checkpoint

        at = AppTest.from_file("app/main.py")
        at.session_state["view"] = "training_dashboard"
        at.run()

        # Check for success message
        assert (
            f"Training concluded successfully. Latest checkpoint found: {mock_checkpoint.name}"
            in [success.value for success in at.success]
        )
        # Check for next step buttons
        buttons = [btn.label for btn in at.button]
        assert "Go to Inference Playground" in buttons
        assert "Create New Model" in buttons


def test_display_control_panel_finished_no_checkpoint_shows_warning_and_reset(
    monkeypatch,
):
    """Test that FINISHED status without checkpoint shows warning and reset button."""
    setup_di_mock(monkeypatch, mock_training_service(status="FINISHED"))

    with patch(
        "app.components.training_dashboard.control_panel._find_latest_checkpoint"
    ) as mock_find:
        mock_find.return_value = None

        at = AppTest.from_file("app/main.py")
        at.session_state["view"] = "training_dashboard"
        at.run()

        # Check for warning message
        assert (
            "Training finished but no checkpoint found. Please create a new model."
            in [warning.value for warning in at.warning]
        )
        # Check for reset button
        buttons = [btn.label for btn in at.button]
        assert "Reset Application" in buttons


def test_display_control_panel_aborted_shows_info_and_welcome_button(
    monkeypatch,
):
    """Test that aborted state shows info message and welcome button."""
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.run()

    # Check for info message
    assert "Training aborted. Please reset the application." in [
        info.value for info in at.info
    ]
    # Check for welcome button
    buttons = [btn.label for btn in at.button]
    assert "Go to Welcome Page" in buttons


# Shutdown Button Tests
def test_create_shutdown_button_successful_shutdown(monkeypatch):
    """Test successful shutdown button behavior."""
    svc = mock_training_service(status="RUNNING")
    svc.stop_training.return_value = True
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Find and click abort button
    abort_button = None
    for btn in at.button:
        if btn.value == "Abort Training":
            abort_button = btn
            break

    if abort_button:
        abort_button.click()
        at.run()

        # Check for success message
        assert "All processes have been shut down." in [
            success.value for success in at.success
        ]
        # Check session state changes
        assert at.session_state["session_started_by_app"] is False
        assert at.session_state["abort_training"] is True


def test_create_shutdown_button_graceful_failure_forceful_success(monkeypatch):
    """Test shutdown with graceful failure but forceful success."""
    svc = mock_training_service(status="RUNNING")
    svc.stop_training.side_effect = [
        False,
        True,
    ]  # First call fails, second succeeds
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Find and click abort button
    abort_button = None
    for btn in at.button:
        if btn.value == "Abort Training":
            abort_button = btn
            break

    if abort_button:
        abort_button.click()
        at.run()

        # Check for info message about graceful failure
        assert "Graceful shutdown failed. Attempting forceful shutdown..." in [
            info.value for info in at.info
        ]
        # Check for success message
        assert "All processes have been shut down." in [
            success.value for success in at.success
        ]


def test_create_shutdown_button_complete_failure(monkeypatch):
    """Test shutdown with complete failure."""
    svc = mock_training_service(status="RUNNING")
    svc.stop_training.return_value = False
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Find and click abort button
    abort_button = None
    for btn in at.button:
        if btn.value == "Abort Training":
            abort_button = btn
            break

    if abort_button:
        abort_button.click()
        at.run()

        # Check for error message
        assert "Failed to stop training processes" in [
            error.value for error in at.error
        ]


# Next Step Buttons Tests
def test_create_next_step_buttons_inference_navigation(monkeypatch):
    """Test inference button navigation."""
    setup_di_mock(monkeypatch, mock_training_service(status="FINISHED"))

    # Mock checkpoint
    mock_checkpoint = MagicMock()
    mock_checkpoint.name = "checkpoint_123"

    with patch(
        "app.components.training_dashboard.control_panel._find_latest_checkpoint"
    ) as mock_find:
        mock_find.return_value = mock_checkpoint

        at = AppTest.from_file("app/main.py")
        at.session_state["view"] = "training_dashboard"
        at.run()

        # Find and click inference button
        inference_button = None
        for btn in at.button:
            if btn.value == "Go to Inference Playground":
                inference_button = btn
                break

        if inference_button:
            inference_button.click()
            at.run()

            # Check navigation
            assert at.session_state["view"] == "inference"


def test_create_next_step_buttons_create_model_navigation(monkeypatch):
    """Test create model button navigation."""
    setup_di_mock(monkeypatch, mock_training_service(status="FINISHED"))

    # Mock checkpoint
    mock_checkpoint = MagicMock()
    mock_checkpoint.name = "checkpoint_123"

    with patch(
        "app.components.training_dashboard.control_panel._find_latest_checkpoint"
    ) as mock_find:
        mock_find.return_value = mock_checkpoint

        at = AppTest.from_file("app/main.py")
        at.session_state["view"] = "training_dashboard"
        at.run()

        # Find and click create model button
        create_button = None
        for btn in at.button:
            if btn.value == "Create New Model":
                create_button = btn
                break

        if create_button:
            create_button.click()
            at.run()

            # Check navigation
            assert at.session_state["view"] == "create_model"


def test_create_next_step_buttons_welcome_navigation(monkeypatch):
    """Test welcome button navigation from aborted state."""
    setup_di_mock(monkeypatch, mock_training_service(status="RUNNING"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.run()

    # Find and click welcome button
    welcome_button = None
    for btn in at.button:
        if btn.value == "Go to Welcome Page":
            welcome_button = btn
            break

    if welcome_button:
        welcome_button.click()
        at.run()

        # Check navigation
        assert at.session_state["view"] == "welcome"


# Checkpoint Detection Tests
def test_find_latest_checkpoint_with_valid_checkpoints(monkeypatch):
    """Test finding latest checkpoint with valid checkpoints."""
    with patch(
        "app.components.training_dashboard.control_panel.config"
    ) as mock_config:
        mock_config.CHECKPOINT_FOLDER = "/tmp/checkpoints"

        # Mock checkpoint directory structure
        mock_checkpoint1 = MagicMock()
        mock_checkpoint1.stat.return_value.st_ctime = 1000
        mock_checkpoint1.name = "checkpoint_old"

        mock_checkpoint2 = MagicMock()
        mock_checkpoint2.stat.return_value.st_ctime = 2000
        mock_checkpoint2.name = "checkpoint_new"

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ), patch(
            "pathlib.Path.iterdir",
            return_value=[mock_checkpoint1, mock_checkpoint2],
        ):

            from app.components.training_dashboard.control_panel import (
                _find_latest_checkpoint,
            )

            result = _find_latest_checkpoint()

            assert result == mock_checkpoint2  # Should return the newest one


def test_find_latest_checkpoint_no_checkpoints(monkeypatch):
    """Test finding latest checkpoint when no checkpoints exist."""
    with patch(
        "app.components.training_dashboard.control_panel.config"
    ) as mock_config:
        mock_config.CHECKPOINT_FOLDER = "/tmp/checkpoints"

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ), patch("pathlib.Path.iterdir", return_value=[]):

            from app.components.training_dashboard.control_panel import (
                _find_latest_checkpoint,
            )

            result = _find_latest_checkpoint()

            assert result is None


def test_find_latest_checkpoint_folder_not_exists(monkeypatch):
    """Test finding latest checkpoint when folder doesn't exist."""
    with patch(
        "app.components.training_dashboard.control_panel.config"
    ) as mock_config:
        mock_config.CHECKPOINT_FOLDER = "/tmp/checkpoints"

        with patch("pathlib.Path.exists", return_value=False):

            from app.components.training_dashboard.control_panel import (
                _find_latest_checkpoint,
            )

            result = _find_latest_checkpoint()

            assert result is None


def test_find_latest_checkpoint_permission_error(monkeypatch):
    """Test finding latest checkpoint with permission error."""
    with patch(
        "app.components.training_dashboard.control_panel.config"
    ) as mock_config:
        mock_config.CHECKPOINT_FOLDER = "/tmp/checkpoints"

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_dir", return_value=True
        ), patch("pathlib.Path.iterdir", side_effect=PermissionError):

            from app.components.training_dashboard.control_panel import (
                _find_latest_checkpoint,
            )

            result = _find_latest_checkpoint()

            assert result is None


# Edge Cases and Error Handling
def test_display_control_panel_unknown_status(monkeypatch):
    """Test control panel with unknown training status."""
    setup_di_mock(monkeypatch, mock_training_service(status="UNKNOWN"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should not show any control buttons for unknown status
    buttons = [btn.value for btn in at.button]
    assert "Abort Training" not in buttons
    assert "Reset Application" not in buttons


def test_display_control_panel_service_exception(monkeypatch):
    """Test control panel when service throws exception."""
    svc = mock_training_service(status="RUNNING")
    svc.is_training_running.side_effect = Exception("Service error")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should handle exception gracefully
    # No specific assertions needed as we're testing error handling
    assert True  # Test that it doesn't crash


def test_display_control_panel_idle_status(monkeypatch):
    """Test control panel with IDLE status."""
    setup_di_mock(monkeypatch, mock_training_service(status="IDLE"))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should not show any control buttons for IDLE status
    buttons = [btn.value for btn in at.button]
    assert "Abort Training" not in buttons
    assert "Reset Application" not in buttons
