# tests/backend/core/test_optimizer.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.core.optimizer as opt_mod
from backend.core.optimizer import Optimizer


@pytest.fixture(autouse=True)
def stub_optax(monkeypatch):
    ada = MagicMock(name="adafactor")
    optax = SimpleNamespace(adafactor=ada)
    monkeypatch.setattr(opt_mod, "optax", optax)
    return optax


@pytest.fixture(autouse=True)
def stub_kd(monkeypatch):
    class _OptimNS:
        def __init__(self):
            self.partial_updates = MagicMock(name="partial_updates")
            self.select = MagicMock(name="select", return_value="MASK")
    kd = SimpleNamespace(optim=_OptimNS())
    monkeypatch.setattr(opt_mod, "kd", kd)
    return kd


def test_create_standard_optimizer_calls_adafactor(stub_optax):
    out = Optimizer.create_standard_optimizer(0.001)
    stub_optax.adafactor.assert_called_once_with(learning_rate=0.001)
    assert out is stub_optax.adafactor.return_value


def test_create_lora_optimizer_wraps_partial_updates(stub_optax, stub_kd):
    base = object()
    stub_optax.adafactor.return_value = base

    out = Optimizer.create_lora_optimizer(0.01)

    stub_optax.adafactor.assert_called_once_with(learning_rate=0.01)
    stub_kd.optim.select.assert_called_once_with("lora")
    stub_kd.optim.partial_updates.assert_called_once_with(base, mask="MASK")
    assert out is stub_kd.optim.partial_updates.return_value