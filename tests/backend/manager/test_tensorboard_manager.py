import time
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from backend.manager.tensorboard_manager import TensorBoardManager
from backend.utils.tensorboard_event_parser import EventFileParser

# --- Test Fixtures ---


@pytest.fixture
def sample_event_data():
    """Sample event data structure returned by EventFileParser."""
    return {
        "metadata": {
            "parameters": {
                "total_params": 1000000,
                "total_bytes": 2000000000,
                "layers": ["layer1", "layer2", "layer3"],
            }
        },
        "training_metrics": {
            "losses/loss": pd.DataFrame(
                {"step": [1, 2, 3, 4, 5], "value": [2.5, 2.1, 1.8, 1.6, 1.4]}
            ),
            "perf_stats/steps_per_sec": pd.DataFrame(
                {
                    "step": [1, 2, 3, 4, 5],
                    "value": [10.0, 10.5, 11.0, 11.2, 11.5],
                }
            ),
            "perf_stats/total_training_time_hours": pd.DataFrame(
                {"step": [1, 2, 3, 4, 5], "value": [0.1, 0.2, 0.3, 0.4, 0.5]}
            ),
            "perf_stats/data_points_per_sec_global": pd.DataFrame(
                {
                    "step": [1, 2, 3, 4, 5],
                    "value": [100.0, 105.0, 110.0, 112.0, 115.0],
                }
            ),
            "perf_stats/train/avg_time_sec": pd.DataFrame(
                {
                    "step": [1, 2, 3, 4, 5],
                    "value": [0.1, 0.095, 0.091, 0.089, 0.087],
                }
            ),
            "perf_stats/evals_along_train/avg_time_sec": pd.DataFrame(
                {
                    "step": [1, 2, 3, 4, 5],
                    "value": [0.05, 0.048, 0.045, 0.043, 0.041],
                }
            ),
        },
        "latest_training_metrics": {
            "perf_stats/steps_per_sec": 11.5,
            "losses/loss": 1.4,
            "perf_stats/total_training_time_hours": 0.5,
            "perf_stats/data_points_per_sec_global": 115.0,
            "perf_stats/train/avg_time_sec": 0.087,
            "perf_stats/evals_along_train/avg_time_sec": 0.041,
        },
    }


@pytest.fixture
def empty_event_data():
    """Empty event data for edge cases."""
    return {}


@pytest.fixture
def tensorboard_manager():
    """Create a TensorBoardManager instance."""
    return TensorBoardManager()


# --- Initialization Tests ---


def test_tensorboard_manager_init(tensorboard_manager):
    """Test TensorBoardManager initialization."""
    assert tensorboard_manager._event_data == {}
    assert tensorboard_manager.tensorboard_log_dir is None
    assert isinstance(tensorboard_manager.event_file_parser, EventFileParser)


def test_tensorboard_manager_inherits_from_base_manager(tensorboard_manager):
    """Test that TensorBoardManager inherits from BaseManager."""
    from backend.manager.base_manager import BaseManager

    assert isinstance(tensorboard_manager, BaseManager)


# --- Work Directory Tests ---


def test_set_work_dir(tensorboard_manager):
    """Test setting work directory."""
    work_dir = "/tmp/test_work_dir"

    with patch("os.makedirs") as mock_makedirs:
        tensorboard_manager.set_work_dir(work_dir)

    assert tensorboard_manager.work_dir == work_dir
    assert tensorboard_manager.tensorboard_log_dir == work_dir
    mock_makedirs.assert_called_once_with(work_dir, exist_ok=True)


def test_set_work_dir_none(tensorboard_manager):
    """Test setting work directory to None."""
    tensorboard_manager.set_work_dir(None)

    assert tensorboard_manager.work_dir is None
    assert tensorboard_manager.tensorboard_log_dir is None


def test_set_work_dir_creates_directory_if_not_exists(tensorboard_manager):
    """Test that work directory is created if it doesn't exist."""
    work_dir = "/tmp/nonexistent_dir"

    with patch("os.makedirs") as mock_makedirs:
        tensorboard_manager.set_work_dir(work_dir)

    mock_makedirs.assert_called_once_with(work_dir, exist_ok=True)


# --- KPI Data Tests ---


@patch.object(EventFileParser, "load_event_data")
def test_get_kpi_data_success(
    mock_load_data, tensorboard_manager, sample_event_data
):
    """Test successful KPI data retrieval."""
    mock_load_data.return_value = sample_event_data

    result = tensorboard_manager.get_kpi_data(total_steps=100)

    # Verify metadata
    assert result["total_params"] == 1000000
    assert result["total_bytes"] == 2000000000
    assert result["total_layers"] == 3

    # Verify progress KPIs
    assert result["current_step"] == 5
    assert result["total_steps"] == 100
    assert result["current_loss"] == 1.4
    assert result["training_time"] == 0.5

    # Verify performance KPIs
    assert result["training_speed"] == 11.5
    assert result["data_throughput"] == 115.0
    assert result["avg_step_time"] == 0.087
    assert result["avg_eval_time"] == 0.041

    # Verify ETA calculation
    assert result["eta_str"] != "N/A"


@patch.object(EventFileParser, "load_event_data")
def test_get_kpi_data_empty_data(
    mock_load_data, tensorboard_manager, empty_event_data
):
    """Test KPI data retrieval with empty event data."""
    mock_load_data.return_value = empty_event_data

    result = tensorboard_manager.get_kpi_data(total_steps=100)

    assert result == {}


@patch.object(EventFileParser, "load_event_data")
def test_get_kpi_data_missing_metrics(mock_load_data, tensorboard_manager):
    """Test KPI data retrieval with missing metrics."""
    incomplete_data = {
        "metadata": {
            "parameters": {
                "total_params": 1000000,
                "total_bytes": 2000000000,
                "layers": ["layer1"],
            }
        },
        "training_metrics": {
            "losses/loss": pd.DataFrame({"step": [1, 2], "value": [2.5, 2.1]})
        },
        "latest_training_metrics": {
            "perf_stats/steps_per_sec": 10.0,
            "losses/loss": 2.1,
        },
    }
    mock_load_data.return_value = incomplete_data

    result = tensorboard_manager.get_kpi_data(total_steps=100)

    # Should handle missing metrics gracefully
    assert result["total_params"] == 1000000
    assert result["total_layers"] == 1
    assert result["current_step"] == 2
    assert result["current_loss"] == 2.1
    assert result["training_speed"] == 10.0
    # Missing metrics should default to 0.0
    assert result["training_time"] == 0.0
    assert result["data_throughput"] == 0.0


@patch.object(EventFileParser, "load_event_data")
def test_get_kpi_data_empty_loss_dataframe(mock_load_data, tensorboard_manager):
    """Test KPI data retrieval with empty loss DataFrame."""
    data_with_empty_loss = {
        "metadata": {
            "parameters": {
                "total_params": 1000000,
                "total_bytes": 2000000000,
                "layers": ["layer1"],
            }
        },
        "training_metrics": {
            "losses/loss": pd.DataFrame(),  # Empty DataFrame
            "perf_stats/steps_per_sec": pd.DataFrame(
                {"step": [1], "value": [10.0]}
            ),
        },
        "latest_training_metrics": {"perf_stats/steps_per_sec": 10.0},
    }
    mock_load_data.return_value = data_with_empty_loss

    result = tensorboard_manager.get_kpi_data(total_steps=100)

    # Should handle empty DataFrame gracefully
    assert result["current_step"] == 0
    assert result["current_loss"] == 0.0


# --- Training Metrics Tests ---


@patch.object(EventFileParser, "load_event_data")
def test_get_training_metrics_success(
    mock_load_data, tensorboard_manager, sample_event_data
):
    """Test successful training metrics retrieval."""
    mock_load_data.return_value = sample_event_data

    result = tensorboard_manager.get_training_metrics()

    assert "losses/loss" in result
    assert "perf_stats/steps_per_sec" in result
    assert isinstance(result["losses/loss"], pd.DataFrame)
    assert len(result["losses/loss"]) == 5


@patch.object(EventFileParser, "load_event_data")
def test_get_training_metrics_empty_data(
    mock_load_data, tensorboard_manager, empty_event_data
):
    """Test training metrics retrieval with empty event data."""
    mock_load_data.return_value = empty_event_data

    result = tensorboard_manager.get_training_metrics()

    assert result == {}


# --- ETA Calculation Tests ---


def test_get_eta_str_normal_case(tensorboard_manager):
    """Test ETA calculation with normal values."""
    result = tensorboard_manager._get_eta_str(
        total_steps=100,
        training_speed=10.0,  # 10 steps per second
        current_step=50,
    )

    # 50 steps remaining at 10 steps/sec = 5 seconds
    assert result == "00:00:05"


def test_get_eta_str_zero_training_speed(tensorboard_manager):
    """Test ETA calculation with zero training speed."""
    result = tensorboard_manager._get_eta_str(
        total_steps=100, training_speed=0.0, current_step=50
    )

    assert result == "N/A"


def test_get_eta_str_current_step_equals_total(tensorboard_manager):
    """Test ETA calculation when current step equals total steps."""
    result = tensorboard_manager._get_eta_str(
        total_steps=100, training_speed=10.0, current_step=100
    )

    assert result == "N/A"


def test_get_eta_str_current_step_greater_than_total(tensorboard_manager):
    """Test ETA calculation when current step exceeds total steps."""
    result = tensorboard_manager._get_eta_str(
        total_steps=100, training_speed=10.0, current_step=150
    )

    assert result == "N/A"


def test_get_eta_str_large_time_values(tensorboard_manager):
    """Test ETA calculation with large time values."""
    result = tensorboard_manager._get_eta_str(
        total_steps=10000,
        training_speed=0.1,  # 0.1 steps per second
        current_step=5000,
    )

    # 5000 steps remaining at 0.1 steps/sec = 50000 seconds = ~13.9 hours
    # Should format as HH:MM:SS
    assert ":" in result
    assert len(result.split(":")) == 3


# --- Cleanup Tests ---


def test_cleanup(tensorboard_manager):
    """Test cleanup method."""
    # Set some data
    tensorboard_manager._event_data = {"test": "data"}

    tensorboard_manager.cleanup()

    assert tensorboard_manager._event_data == {}


# --- Integration Tests ---


@patch.object(EventFileParser, "load_event_data")
def test_full_workflow(mock_load_data, tensorboard_manager, sample_event_data):
	"""Test complete workflow from setting work dir to getting KPI data."""
	mock_load_data.return_value = sample_event_data

	# Set work directory
	work_dir = "/tmp/test_work_dir"
	with patch("os.makedirs"):
		tensorboard_manager.set_work_dir(work_dir)

	# Get KPI data
	result = tensorboard_manager.get_kpi_data(total_steps=100)

	# Verify the workflow worked
	assert result["total_params"] == 1000000
	assert result["current_step"] == 5
	assert result["eta_str"] != "N/A"

	# Verify EventFileParser was called
	mock_load_data.assert_called_once()


# --- Error Handling Tests ---


@patch.object(EventFileParser, "load_event_data")
def test_get_kpi_data_parser_exception(mock_load_data, tensorboard_manager):
    """Test KPI data retrieval when EventFileParser raises an exception."""
    mock_load_data.side_effect = Exception("Parser error")

    # Should handle exceptions gracefully
    with pytest.raises(ValueError, match="Error loading event data: Parser error"):
        tensorboard_manager.get_kpi_data(total_steps=100)


@patch.object(EventFileParser, "load_event_data")
def test_get_training_metrics_parser_exception(
    mock_load_data, tensorboard_manager
):
    """Test training metrics retrieval when EventFileParser raises an exception."""
    mock_load_data.side_effect = Exception("Parser error")

    # Should handle exceptions gracefully
    with pytest.raises(ValueError, match="Error loading event data: Parser error"):
        tensorboard_manager.get_training_metrics()
