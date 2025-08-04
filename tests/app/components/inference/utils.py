from unittest.mock import MagicMock

import pytest

@pytest.fixture
def mock_inferencer(monkeypatch):
    """Mock Inferencer for checkpoint selection.(avoid heavy imports)"""
    mock_inferencer = MagicMock()
    mock_inferencer.list_checkpoints.return_value = [
        "checkpoint1",
        "checkpoint2",
        "checkpoint3",
    ]
    mock_inferencer.is_loaded.return_value = True
    mock_inferencer.load_model.return_value = True
    mock_inferencer.delete_checkpoint.return_value = True

    def mock_inferencer_func(work_dir):
        return mock_inferencer

    monkeypatch.setattr(
        "backend.inferencer.Inferencer",
        mock_inferencer_func,
    )
    return mock_inferencer