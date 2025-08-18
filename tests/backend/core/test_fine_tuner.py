# tests/backend/core/test_fine_tuner.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.core.fine_tuner as ft_mod
from backend.core.fine_tuner import Trainer, StandardTrainer, LoRATrainer, QuantizationAwareTrainer


@pytest.fixture(autouse=True)
def stub_kd(monkeypatch):
	class _TrainerStub:
		def __init__(self, **kwargs):
			self.kwargs = kwargs
	class _CKPTS:
		def __init__(self):
			self.Checkpointer = MagicMock(name="Checkpointer", side_effect=lambda **kw: SimpleNamespace(kw=kw))
	kd = SimpleNamespace(
		train=SimpleNamespace(Trainer=lambda **kw: _TrainerStub(**kw)),
		ckpts=_CKPTS(),
		data=SimpleNamespace(Pipeline=object),
		optim=SimpleNamespace(),
	)
	monkeypatch.setattr(ft_mod, "kd", kd)
	return kd


def _tc(method="Standard", lr=0.1, epochs=7, variant="GemmaX", lora_rank=4):
	params = SimpleNamespace(lora_rank=lora_rank)
	mc = SimpleNamespace(model_variant=variant, learning_rate=lr, epochs=epochs, method=method, parameters=params)
	return SimpleNamespace(model_config=mc)


def test_base_trainer_constructs_kd_trainer(stub_kd):
	t = Trainer()
	model, init_t, optimizer = object(), object(), object()
	train_ds, workdir, steps = object(), "/tmp/work", 5

	out = t.create_trainer(model, init_t, optimizer, train_ds, workdir, steps)

	assert out.kwargs["workdir"] == "/tmp/work"
	assert out.kwargs["train_ds"] is train_ds
	assert out.kwargs["model"] is model
	assert out.kwargs["init_transform"] is init_t
	assert out.kwargs["num_train_steps"] == 5
	assert "train_losses" in out.kwargs
	assert "optimizer" in out.kwargs
	assert out.kwargs["checkpointer"].kw["save_interval_steps"] == 5


def test_standard_trainer_calls_components(monkeypatch):
	tc = _tc(method="Standard", lr=0.001, epochs=3, variant="G")
	monkeypatch.setattr(ft_mod.Model, "create_standard_model", MagicMock(return_value="M"))
	monkeypatch.setattr(ft_mod.Checkpoint, "create_standard_checkpoint", MagicMock(return_value="C"))
	monkeypatch.setattr(ft_mod.Optimizer, "create_standard_optimizer", MagicMock(return_value="O"))
	base = StandardTrainer()

	out = base.create_trainer(tc, train_ds="DS", workdir="/w")

	ft_mod.Model.create_standard_model.assert_called_once_with("G")
	ft_mod.Checkpoint.create_standard_checkpoint.assert_called_once_with("G")
	ft_mod.Optimizer.create_standard_optimizer.assert_called_once_with(0.001)
	assert out.kwargs["num_train_steps"] == 3


def test_lora_trainer_calls_components(monkeypatch):
	tc = _tc(method="LoRA", lr=0.01, epochs=9, variant="G", lora_rank=8)
	monkeypatch.setattr(ft_mod.Model, "create_lora_model", MagicMock(return_value="M"))
	monkeypatch.setattr(ft_mod.Checkpoint, "create_lora_checkpoint", MagicMock(return_value="C"))
	monkeypatch.setattr(ft_mod.Optimizer, "create_lora_optimizer", MagicMock(return_value="O"))
	base = LoRATrainer()

	out = base.create_trainer(tc, train_ds="DS", workdir="/w")

	ft_mod.Model.create_lora_model.assert_called_once_with("G", 8)
	ft_mod.Checkpoint.create_lora_checkpoint.assert_called_once_with("G")
	ft_mod.Optimizer.create_lora_optimizer.assert_called_once_with(0.01)
	assert out.kwargs["num_train_steps"] == 9


def test_qat_trainer_calls_components(monkeypatch):
	tc = _tc(method="QuantizationAware", lr=0.02, epochs=4, variant="G")
	monkeypatch.setattr(ft_mod.Model, "create_quantization_aware_model", MagicMock(return_value="M"))
	monkeypatch.setattr(ft_mod.Checkpoint, "create_standard_checkpoint", MagicMock(return_value="C"))
	monkeypatch.setattr(ft_mod.Optimizer, "create_standard_optimizer", MagicMock(return_value="O"))
	base = QuantizationAwareTrainer()

	out = base.create_trainer(tc, train_ds="DS", workdir="/w")

	ft_mod.Model.create_quantization_aware_model.assert_called_once_with("G")
	ft_mod.Checkpoint.create_standard_checkpoint.assert_called_once_with("G")
	ft_mod.Optimizer.create_standard_optimizer.assert_called_once_with(0.02)
	assert out.kwargs["num_train_steps"] == 4