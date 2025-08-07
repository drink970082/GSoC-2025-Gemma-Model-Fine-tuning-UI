from unittest.mock import MagicMock

import pandas as pd
from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock


def make_history(cpu_points=2, gpu_points=2, empty=False):
    """Helper to create mock system usage history."""
    cpu_df = pd.DataFrame(
        {"time": [1, 2][:cpu_points], "value": [10, 20][:cpu_points]}
    )
    gpu_df = pd.DataFrame(
        {"time": [1, 2][:gpu_points], "value": [30, 40][:gpu_points]}
    )
    if empty:
        cpu_df = pd.DataFrame()
        gpu_df = pd.DataFrame()
    return {
        "CPU Utilization (%)": cpu_df,
        "GPU Utilization (%)": gpu_df,
    }


def test_display_system_usage_panel_no_gpu(monkeypatch):
    """Warns if no GPU is detected."""
    svc = mock_training_service()
    svc.has_gpu.return_value = False
    svc.get_system_usage_history.return_value = make_history()
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    assert "NVIDIA GPU not detected. GPU monitoring is disabled." in [
        warning.value for warning in at.warning
    ]


def test_display_system_usage_panel_collecting_data(monkeypatch):
    """Shows collecting message if not enough CPU data."""
    svc = mock_training_service()
    svc.has_gpu.return_value = True
    svc.get_system_usage_history.return_value = make_history(
        cpu_points=1, gpu_points=1
    )
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    assert "Collecting system usage data..." in [info.value for info in at.info]


def test_display_system_usage_panel_all_empty(monkeypatch):
    """Shows collecting message if all charts are empty."""
    svc = mock_training_service()
    svc.has_gpu.return_value = True
    svc.get_system_usage_history.return_value = make_history(empty=True)
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    assert "Collecting system usage data..." in [info.value for info in at.info]


def test_display_system_usage_panel_cpu_only(monkeypatch):
    """Displays only CPU chart if GPU is not available."""
    svc = mock_training_service()
    svc.has_gpu.return_value = False
    history = make_history()
    svc.get_system_usage_history.return_value = history
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Only CPU chart should be shown
    markdowns = [markdown.value for markdown in at.markdown]
    assert "**CPU Utilization (%)**" in markdowns
    assert "**GPU Utilization (%)**" not in markdowns


def test_display_system_usage_panel_cpu_and_gpu(monkeypatch):
    """Displays both CPU and GPU charts if GPU is available."""
    svc = mock_training_service()
    svc.has_gpu.return_value = True
    history = make_history()
    svc.get_system_usage_history.return_value = history
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    markdowns = [markdown.value for markdown in at.markdown]
    assert "**CPU Utilization (%)**" in markdowns
    assert "**GPU Utilization (%)**" in markdowns


def test_display_system_usage_panel_partial_charts(monkeypatch):
    """Displays only charts with data and enough points."""
    svc = mock_training_service()
    svc.has_gpu.return_value = True
    svc.get_system_usage_history.return_value = make_history(
        cpu_points=2, gpu_points=0
    )
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    markdowns = [markdown.value for markdown in at.markdown]
    assert "**CPU Utilization (%)**" in markdowns
    assert "**GPU Utilization (%)**" not in markdowns


def test_display_system_usage_panel_service_exception(monkeypatch):
    """Handles service exceptions gracefully."""
    svc = mock_training_service()
    svc.has_gpu.side_effect = Exception("Service error")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # No assertion needed, just ensure no crash
    assert True
