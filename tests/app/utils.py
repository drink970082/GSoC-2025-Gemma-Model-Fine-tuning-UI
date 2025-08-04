from unittest.mock import MagicMock

import pytest

from config.dataclass import DataConfig, ModelConfig, TrainingConfig

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


def get_default_config() -> TrainingConfig:
    """Default training configuration for testing."""
    return TrainingConfig(
        model_name="gemma-3-1b-test",
        data_config=DataConfig(
            source="tensorflow",
            dataset_name="mtnt",
            split="train",
            shuffle=True,
            batch_size=4,
            seq2seq_in_prompt="src",
            seq2seq_in_response="dst",
            seq2seq_max_length=200,
            seq2seq_truncate=True,
            config=None,
        ),
        model_config=ModelConfig(
            model_variant="Gemma3_1B",
            epochs=1,
            learning_rate=1e-4,
            method="Standard",
        ),
    )
