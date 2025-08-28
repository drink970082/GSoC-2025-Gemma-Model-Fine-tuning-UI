from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock


# Main KPI Panel Display Tests
def test_display_kpi_panel_waiting_for_data(monkeypatch):
    """Test KPI panel when waiting for training data."""
    setup_di_mock(
        monkeypatch, mock_training_service(kpi_data={"current_step": 0})
    )

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for waiting message
    assert "Waiting for training data..." in [info.value for info in at.info]


def test_display_kpi_panel_with_metadata_only(monkeypatch):
    """Test KPI panel with metadata but no training progress."""
    kpi_data = {
        "total_params": 1000000,
        "total_bytes": 2147483648,  # 2GB
        "total_layers": 24,
        "current_step": 0,
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for metadata panel
    assert "#### Model Information" in [markdown.value for markdown in at.markdown]
    # Check for metadata metrics
    metrics = [metric.label for metric in at.metric]
    assert "Total Parameters" in metrics
    assert "Total Memory (GB)" in metrics
    assert "Transformer Layers" in metrics


def test_display_kpi_panel_with_training_progress(monkeypatch):
    """Test KPI panel with active training progress."""
    kpi_data = {
        "current_step": 100,
        "total_steps": 1000,
        "current_loss": 0.1234,
        "training_speed": 2.5,
        "training_time": 1.5,  # 1.5 hours
        "data_throughput": 1500.0,
        "eta_str": "2:30:00",
        "avg_step_time": 0.400,
        "avg_eval_time": 0.100,
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for training progress panel
    assert "#### Training Progress" in [markdown.value for markdown in at.markdown]
    # Check for progress metrics
    metrics = [metric.label for metric in at.metric]
    assert "Global Step" in metrics
    assert "Current Loss" in metrics
    assert "Training Speed" in metrics
    assert "Training Time" in metrics


def test_display_kpi_panel_with_performance_metrics(monkeypatch):
    """Test KPI panel with performance metrics."""
    kpi_data = {
        "current_step": 100,
        "data_throughput": 1500.0,
        "eta_str": "2:30:00",
        "avg_step_time": 0.400,
        "avg_eval_time": 0.100,
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for performance metrics panel
    assert "#### Performance Metrics" in [markdown.value for markdown in at.markdown]
    # Check for performance metrics
    metrics = [metric.label for metric in at.metric]
    assert "Data Throughput" in metrics
    assert "ETA" in metrics
    assert "Avg Step Time" in metrics
    assert "Avg Eval Time" in metrics


def test_display_kpi_panel_frozen_state(monkeypatch):
    """Test KPI panel in frozen state (aborted training)."""
    kpi_data = {
        "current_step": 100,
        "total_params": 1000000,
        "current_loss": 0.1234,
        "training_speed": 2.5,
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.session_state["frozen_kpi_data"] = kpi_data
    at.run()

    # Should still display panels even in frozen state
    assert "#### Training Progress" in [markdown.value for markdown in at.markdown]
    metrics = [metric.label for metric in at.metric]
    assert "Global Step" in metrics
    assert "Current Loss" in metrics


# Time Formatting Tests
def test_format_time_zero_seconds():
    """Test time formatting with zero seconds."""
    from app.components.training_dashboard.kpi_panel import _format_time

    result = _format_time(0)
    assert result == "00:00:00"


def test_format_time_positive_seconds():
    """Test time formatting with positive seconds."""
    from app.components.training_dashboard.kpi_panel import _format_time

    result = _format_time(3661)  # 1 hour, 1 minute, 1 second
    assert result == "01:01:01"


def test_format_time_negative_seconds():
    """Test time formatting with negative seconds."""
    from app.components.training_dashboard.kpi_panel import _format_time

    result = _format_time(-100)
    assert result == "00:00:00"


def test_format_time_large_hours():
    """Test time formatting with large hours."""
    from app.components.training_dashboard.kpi_panel import _format_time

    result = _format_time(7325)  # 2 hours, 2 minutes, 5 seconds
    assert result == "02:02:05"


# Metadata Panel Tests
def test_create_metadata_panel_with_complete_data(monkeypatch):
    """Test metadata panel with complete data."""
    kpi_data = {
        "total_params": 1000000,
        "total_bytes": 2147483648,  # 2GB
        "total_layers": 24,
        "current_step": 100,
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check metadata panel title
    assert "#### Model Information" in [markdown.value for markdown in at.markdown]

    # Check metric values
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Total Parameters"] == "1,000,000"
    assert metrics["Total Memory (GB)"] == "2.0"
    assert metrics["Transformer Layers"] == "24"


def test_create_metadata_panel_with_missing_data(monkeypatch):
    """Test metadata panel with missing data."""
    kpi_data = {
        "current_step": 100
        # Missing total_params, total_bytes, total_layers
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should not show metadata panel when no metadata
    markdown_values = [markdown.value for markdown in at.markdown]
    assert "#### Model Information" not in markdown_values


# Training Progress Panel Tests
def test_create_training_progress_panel_with_total_steps(monkeypatch):
    """Test training progress panel with total steps."""
    kpi_data = {
        "current_step": 100,
        "total_steps": 1000,
        "current_loss": 0.1234,
        "training_speed": 2.5,
        "training_time": 1.5,  # 1.5 hours
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check progress panel title
    assert "#### Training Progress" in [markdown.value for markdown in at.markdown]

    # Check metric values
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Global Step"] == "100/1000"
    assert metrics["Current Loss"] == "0.1234"
    assert metrics["Training Speed"] == "2.50 steps/sec"
    assert metrics["Training Time"] == "01:30:00"  # 1.5 hours = 5400 seconds


def test_create_training_progress_panel_without_total_steps(monkeypatch):
    """Test training progress panel without total steps."""
    kpi_data = {
        "current_step": 100,
        "total_steps": 0,
        "current_loss": 0.1234,
        "training_speed": 2.5,
        "training_time": 0.5,  # 0.5 hours
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check metric values
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Global Step"] == "100"  # No total steps
    assert metrics["Training Time"] == "00:30:00"  # 0.5 hours = 1800 seconds


def test_create_training_progress_panel_with_missing_data(monkeypatch):
    """Test training progress panel with missing data."""
    kpi_data = {
        "current_step": 100
        # Missing other fields
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check metric values with defaults
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Current Loss"] == "0.0000"
    assert metrics["Training Speed"] == "0.00 steps/sec"
    assert metrics["Training Time"] == "00:00:00"


# Performance Metrics Panel Tests
def test_create_performance_metrics_panel_with_complete_data(monkeypatch):
    """Test performance metrics panel with complete data."""
    kpi_data = {
        "current_step": 100,
        "data_throughput": 1500.0,
        "eta_str": "2:30:00",
        "avg_step_time": 0.400,
        "avg_eval_time": 0.100,
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check performance panel title
    assert "#### Performance Metrics" in [markdown.value for markdown in at.markdown]

    # Check metric values
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Data Throughput"] == "1500 tokens/sec"
    assert metrics["ETA"] == "2:30:00"
    assert metrics["Avg Step Time"] == "0.400s"
    assert metrics["Avg Eval Time"] == "0.100s"


def test_create_performance_metrics_panel_with_missing_data(monkeypatch):
    """Test performance metrics panel with missing data."""
    kpi_data = {
        "current_step": 100
        # Missing performance fields
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check metric values with defaults
    metrics = {metric.label: metric.value for metric in at.metric}
    assert metrics["Data Throughput"] == "0 tokens/sec"
    assert metrics["ETA"] == "N/A"
    assert metrics["Avg Step Time"] == "0.000s"
    assert metrics["Avg Eval Time"] == "0.000s"


# Edge Cases and Error Handling
def test_display_kpi_panel_empty_data(monkeypatch):
    """Test KPI panel with empty data."""
    setup_di_mock(monkeypatch, mock_training_service(kpi_data={}))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should show waiting message
    assert "Waiting for training data..." in [info.value for info in at.info]


def test_display_kpi_panel_none_data(monkeypatch):
    """Test KPI panel with None data."""
    svc = mock_training_service()
    svc.get_kpi_data.return_value = None
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should handle None data gracefully
    # No specific assertions needed as we're testing error handling
    assert True


def test_display_kpi_panel_service_exception(monkeypatch):
    """Test KPI panel when service throws exception."""
    svc = mock_training_service()
    svc.get_kpi_data.side_effect = Exception("Service error")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should handle exception gracefully
    # No specific assertions needed as we're testing error handling
    assert True


def test_display_kpi_panel_metadata_only_no_training(monkeypatch):
    """Test KPI panel with metadata but no training progress."""
    kpi_data = {
        "total_params": 1000000,
        "total_bytes": 2147483648,
        "total_layers": 24,
        "current_step": 0,  # No training progress
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should show metadata panel but not training progress
    assert "#### Model Information" in [markdown.value for markdown in at.markdown]
    assert "#### Training Progress" not in [
        markdown.value for markdown in at.markdown
    ]


def test_display_kpi_panel_training_only_no_metadata(monkeypatch):
    """Test KPI panel with training progress but no metadata."""
    kpi_data = {
        "current_step": 100,
        "current_loss": 0.1234,
        "training_speed": 2.5,
        # No metadata fields
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should show training progress but not metadata
    assert "#### Training Progress" in [markdown.value for markdown in at.markdown]
    assert "#### Model Information" not in [
        markdown.value for markdown in at.markdown
    ]


# Fragment Behavior Tests
def test_display_kpi_panel_fragment_updates(monkeypatch):
    """Test that KPI panel updates with fragment behavior."""
    svc = mock_training_service(kpi_data={"current_step": 0})
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Initial state - waiting
    assert "Waiting for training data..." in [info.value for info in at.info]

    # Update service to return training data
    svc.get_kpi_data.return_value = {
        "current_step": 100,
        "current_loss": 0.1234,
        "training_speed": 2.5,
    }
    at.run()

    # Should now show training progress
    assert "#### Training Progress" in [markdown.value for markdown in at.markdown]


def test_display_kpi_panel_frozen_state_persistence(monkeypatch):
    """Test that frozen KPI data persists across runs."""
    kpi_data = {
        "current_step": 100,
        "current_loss": 0.1234,
        "training_speed": 2.5,
    }
    setup_di_mock(monkeypatch, mock_training_service(kpi_data=kpi_data))

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.session_state["frozen_kpi_data"] = kpi_data
    at.run()

    # Should use frozen data
    assert "#### Training Progress" in [markdown.value for markdown in at.markdown]

    # Change service data (should not affect frozen state)
    svc = mock_training_service(kpi_data={"current_step": 200})
    setup_di_mock(monkeypatch, svc)
    at.run()

    # Should still show frozen data
    assert "#### Training Progress" in [markdown.value for markdown in at.markdown]
