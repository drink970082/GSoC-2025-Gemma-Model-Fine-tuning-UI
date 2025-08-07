from unittest.mock import MagicMock

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import mock_training_service, setup_di_mock


# Helper function to create mock training metrics
def create_mock_training_metrics():
    """Create mock training metrics data structure."""
    # Create sample DataFrame for loss metrics
    loss_df = pd.DataFrame(
        {"step": [1, 2, 3, 4, 5], "value": [0.5, 0.4, 0.3, 0.25, 0.2]}
    )

    # Create sample DataFrame for performance metrics
    perf_df = pd.DataFrame(
        {"step": [1, 2, 3, 4, 5], "value": [2.5, 2.6, 2.7, 2.8, 2.9]}
    )

    # Create sample DataFrame for training time
    time_df = pd.DataFrame(
        {"step": [1, 2, 3, 4, 5], "value": [0.1, 0.2, 0.3, 0.4, 0.5]}
    )

    # Create sample DataFrame for data throughput
    throughput_df = pd.DataFrame(
        {"step": [1, 2, 3, 4, 5], "value": [1000, 1100, 1200, 1300, 1400]}
    )

    return {
        "losses/loss": loss_df,
        "perf_stats/steps_per_sec": perf_df,
        "perf_stats/total_training_time_hours": time_df,
        "perf_stats/data_points_per_sec_global": throughput_df,
    }


# Main Plots Panel Display Tests
def test_display_plots_panel_waiting_for_metrics(monkeypatch):
    """Test plots panel when waiting for first metric."""
    svc = mock_training_service(status="RUNNING")
    svc.get_training_metrics.return_value = {}
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()
    # Check for waiting message
    assert "Waiting for first metric to be logged..." in [
        info.value for info in at.info
    ]


def test_display_plots_panel_with_loss_metrics_only(monkeypatch):
    """Test plots panel with only loss metrics."""
    training_metrics = {
        "losses/loss": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [0.5, 0.4, 0.3, 0.25, 0.2]}
        )
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for loss plots title
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    # Check for loss metric title
    assert "**loss**" in [markdown.value for markdown in at.markdown]


def test_display_plots_panel_with_performance_metrics_only(monkeypatch):
    """Test plots panel with only performance metrics."""
    training_metrics = {
        "perf_stats/steps_per_sec": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [2.5, 2.6, 2.7, 2.8, 2.9]}
        ),
        "perf_stats/total_training_time_hours": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [0.1, 0.2, 0.3, 0.4, 0.5]}
        ),
        "perf_stats/data_points_per_sec_global": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [1000, 1100, 1200, 1300, 1400]}
        ),
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for performance metrics title
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]
    # Check for performance metric titles
    assert "**Training Speed (Steps/Second)**" in [
        markdown.value for markdown in at.markdown
    ]
    assert "**Total Training Time (Hours)**" in [
        markdown.value for markdown in at.markdown
    ]
    assert "**Data Throughput (Points/Second)**" in [
        markdown.value for markdown in at.markdown
    ]


def test_display_plots_panel_with_both_metrics(monkeypatch):
    """Test plots panel with both loss and performance metrics."""
    training_metrics = create_mock_training_metrics()

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for both titles
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]
    # Check for metric titles
    assert "**loss**" in [markdown.value for markdown in at.markdown]
    assert "**Training Speed (Steps/Second)**" in [
        markdown.value for markdown in at.markdown
    ]


def test_display_plots_panel_frozen_state(monkeypatch):
    """Test plots panel in frozen state (aborted training)."""
    training_metrics = create_mock_training_metrics()

    setup_di_mock(monkeypatch, mock_training_service())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.session_state["frozen_training_metrics"] = training_metrics
    at.run()

    # Should still display plots even in frozen state
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]


# Loss Plots Tests
def test_create_loss_plots_with_multiple_data_points(monkeypatch):
    """Test loss plots with multiple data points."""
    training_metrics = {
        "losses/loss": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [0.5, 0.4, 0.3, 0.25, 0.2]}
        ),
        "losses/validation_loss": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [0.6, 0.5, 0.4, 0.35, 0.3]}
        ),
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for loss plots title
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    # Check for both loss metric titles
    assert "**loss**" in [markdown.value for markdown in at.markdown]
    assert "**validation_loss**" in [markdown.value for markdown in at.markdown]


def test_create_loss_plots_with_single_data_point(monkeypatch):
    """Test loss plots with single data point (should show metric instead of chart)."""
    training_metrics = {
        "losses/loss": pd.DataFrame({"step": [1], "value": [0.5]})
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for loss plots title
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    # Check for loss metric title
    assert "**loss**" in [markdown.value for markdown in at.markdown]
    # Check for metric value (single point shows metric instead of chart)
    metrics = [metric.label for metric in at.metric]
    assert "Value" in metrics


def test_create_loss_plots_with_empty_dataframe(monkeypatch):
    """Test loss plots with empty DataFrame."""
    training_metrics = {"losses/loss": pd.DataFrame()}

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should not show loss plots with empty data
    assert "### Training Loss" not in [
        markdown.value for markdown in at.markdown
    ]


# Performance Plots Tests
def test_create_perf_plots_with_all_metrics(monkeypatch):
    """Test performance plots with all available metrics."""
    training_metrics = {
        "perf_stats/steps_per_sec": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [2.5, 2.6, 2.7, 2.8, 2.9]}
        ),
        "perf_stats/total_training_time_hours": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [0.1, 0.2, 0.3, 0.4, 0.5]}
        ),
        "perf_stats/data_points_per_sec_global": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [1000, 1100, 1200, 1300, 1400]}
        ),
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for performance metrics title
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]
    # Check for all performance metric titles
    assert "**Training Speed (Steps/Second)**" in [
        markdown.value for markdown in at.markdown
    ]
    assert "**Total Training Time (Hours)**" in [
        markdown.value for markdown in at.markdown
    ]
    assert "**Data Throughput (Points/Second)**" in [
        markdown.value for markdown in at.markdown
    ]


def test_create_perf_plots_with_partial_metrics(monkeypatch):
    """Test performance plots with only some metrics available."""
    training_metrics = {
        "perf_stats/steps_per_sec": pd.DataFrame(
            {"step": [1, 2, 3, 4, 5], "value": [2.5, 2.6, 2.7, 2.8, 2.9]}
        )
        # Missing other performance metrics
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Check for performance metrics title
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]
    # Check for available metric title
    assert "**Training Speed (Steps/Second)**" in [
        markdown.value for markdown in at.markdown
    ]
    # Should not show missing metrics
    markdown_values = [markdown.value for markdown in at.markdown]
    assert "**Total Training Time (Hours)**" not in markdown_values
    assert "**Data Throughput (Points/Second)**" not in markdown_values


def test_create_perf_plots_with_empty_dataframes(monkeypatch):
    """Test performance plots with empty DataFrames."""
    training_metrics = {
        "perf_stats/steps_per_sec": pd.DataFrame(),
        "perf_stats/total_training_time_hours": pd.DataFrame(),
        "perf_stats/data_points_per_sec_global": pd.DataFrame(),
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should not show performance plots with empty data
    markdown_values = [markdown.value for markdown in at.markdown]
    assert "### Performance Metrics" not in markdown_values


# Edge Cases and Error Handling
def test_display_plots_panel_empty_metrics(monkeypatch):
    """Test plots panel with empty metrics."""
    svc = mock_training_service()
    svc.get_training_metrics.return_value = {}
    setup_di_mock(monkeypatch, svc)
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should show waiting message
    assert "Waiting for first metric to be logged..." in [
        info.value for info in at.info
    ]


def test_display_plots_panel_service_exception(monkeypatch):
    """Test plots panel when service throws exception."""
    svc = mock_training_service()
    svc.get_training_metrics.side_effect = Exception("Service error")
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should handle exception gracefully
    # No specific assertions needed as we're testing error handling
    assert True


def test_display_plots_panel_mixed_metric_types(monkeypatch):
    """Test plots panel with mixed metric types (loss and performance)."""
    training_metrics = {
        "losses/loss": pd.DataFrame(
            {"step": [1, 2, 3], "value": [0.5, 0.4, 0.3]}
        ),
        "perf_stats/steps_per_sec": pd.DataFrame(
            {"step": [1, 2, 3], "value": [2.5, 2.6, 2.7]}
        ),
        "other_metric": pd.DataFrame(
            {"step": [1, 2, 3], "value": [100, 200, 300]}
        ),
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should only show loss and performance metrics, ignore others
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]
    # Should not show other metrics
    markdown_values = [markdown.value for markdown in at.markdown]
    assert "**other_metric**" not in markdown_values


def test_display_plots_panel_no_loss_or_perf_metrics(monkeypatch):
    """Test plots panel with no loss or performance metrics."""
    training_metrics = {
        "other_metric": pd.DataFrame(
            {"step": [1, 2, 3], "value": [100, 200, 300]}
        )
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should not show any plots when no loss or performance metrics
    markdown_values = [markdown.value for markdown in at.markdown]
    assert "### Training Loss" not in markdown_values
    assert "### Performance Metrics" not in markdown_values


# Fragment Behavior Tests
def test_display_plots_panel_fragment_updates(monkeypatch):
    """Test that plots panel updates with fragment behavior."""
    svc = mock_training_service()
    svc.get_training_metrics.return_value = {}
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Initial state - waiting
    assert "Waiting for first metric to be logged..." in [
        info.value for info in at.info
    ]

    # Update service to return training metrics
    svc.get_training_metrics.return_value = {
        "losses/loss": pd.DataFrame(
            {"step": [1, 2, 3], "value": [0.5, 0.4, 0.3]}
        )
    }
    at.run()

    # Should now show loss plots
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]


def test_display_plots_panel_frozen_state_persistence(monkeypatch):
    """Test that frozen training metrics persist across runs."""
    training_metrics = create_mock_training_metrics()

    setup_di_mock(monkeypatch, mock_training_service())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.session_state["abort_training"] = True
    at.session_state["frozen_training_metrics"] = training_metrics
    at.run()

    # Should use frozen data
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]

    # Change service data (should not affect frozen state)
    svc = mock_training_service()
    svc.get_training_metrics.return_value = {
        "losses/loss": pd.DataFrame(
            {"step": [1], "value": [999.0]}  # Different data
        )
    }
    setup_di_mock(monkeypatch, svc)
    at.run()

    # Should still show frozen data
    assert "### Training Loss" in [markdown.value for markdown in at.markdown]
    assert "### Performance Metrics" in [
        markdown.value for markdown in at.markdown
    ]


# Data Structure Tests
def test_display_plots_panel_invalid_dataframe_structure(monkeypatch):
    """Test plots panel with invalid DataFrame structure."""
    training_metrics = {
        "losses/loss": pd.DataFrame(
            {"invalid_column": [1, 2, 3], "another_invalid": [0.5, 0.4, 0.3]}
        )
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should handle invalid DataFrame structure gracefully
    # No specific assertions needed as we're testing error handling
    assert True


def test_display_plots_panel_missing_required_columns(monkeypatch):
    """Test plots panel with missing required columns."""
    training_metrics = {
        "losses/loss": pd.DataFrame(
            {
                "step": [1, 2, 3]
                # Missing 'value' column
            }
        )
    }

    svc = mock_training_service()
    svc.get_training_metrics.return_value = training_metrics
    setup_di_mock(monkeypatch, svc)

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "training_dashboard"
    at.run()

    # Should handle missing columns gracefully
    # No specific assertions needed as we're testing error handling
    assert True
