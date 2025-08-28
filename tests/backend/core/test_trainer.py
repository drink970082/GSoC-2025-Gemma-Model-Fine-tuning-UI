# tests/backend/core/test_trainer.py
from types import SimpleNamespace
from unittest.mock import MagicMock
import os

import pytest

import backend.core.trainer as tr_mod
from backend.core.trainer import ModelTrainer


def _tc(method="Standard"):
    mc = SimpleNamespace(method=method)
    return SimpleNamespace(model_config=mc, data_config=SimpleNamespace(source="json"))


def test_setup_environment_sets_xla_env():
    mt = ModelTrainer(_tc(), work_dir="/w")
    mt.setup_environment()
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == tr_mod.XLA_MEM_FRACTION


def test_train_happy_path(monkeypatch):
    mt = ModelTrainer(_tc("Standard"), work_dir="/w")

    # create_pipeline returns pipeline with get_train_dataset
    pipeline = SimpleNamespace(get_train_dataset=MagicMock(return_value="DS"))
    monkeypatch.setattr(tr_mod, "create_pipeline", MagicMock(return_value=pipeline))

    # FINE_TUNE_STRATEGIES provides an entry with create_trainer returning trainer obj with train()
    fake_strategy = SimpleNamespace(create_trainer=MagicMock(return_value=SimpleNamespace(train=MagicMock(return_value=("OK", 1)))))
    monkeypatch.setattr(tr_mod, "FINE_TUNE_STRATEGIES", {"Standard": fake_strategy})

    out = mt.train()
    assert out == ("OK", 1)
    tr_mod.create_pipeline.assert_called_once()
    fake_strategy.create_trainer.assert_called_once_with(mt.training_config, "DS", "/w")


def test_train_bubbles_exception_and_prints(monkeypatch, capsys):
    mt = ModelTrainer(_tc("Standard"), work_dir="/w")

    monkeypatch.setattr(tr_mod, "create_pipeline", MagicMock(side_effect=RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        mt.train()

    # stderr contains traceback line(s)
    captured = capsys.readouterr()
    assert "Traceback" in captured.err