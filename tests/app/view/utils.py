from unittest.mock import MagicMock

import pytest


def setup_di_mock(monkeypatch, training_service):
    """Helper to setup DI container mocking."""

    def mock_get_service(name):
        if name == "training_service":
            return training_service
        return MagicMock()

    monkeypatch.setattr("services.di_container.get_service", mock_get_service)


def mock_training_service(
    status: str = "RUNNING",
    kpi_data: dict = {"current_step": 10, "total_params": 1000000},
    log_contents: tuple = ("", ""),
):
    svc = MagicMock()
    svc.is_training_running.return_value = status
    svc.get_kpi_data.return_value = kpi_data
    svc.get_log_contents.return_value = log_contents
    return svc
