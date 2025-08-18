# tests/backend/core/test_checkpoint.py
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.core.checkpoint as ck_mod
from backend.core.checkpoint import Checkpoint


@pytest.fixture(autouse=True)
def stub_gemma(monkeypatch):
	# Build gm stub
	class _CKPTS:
		class CheckpointPath:
			DUMMYMODEL_IT = "/ckpts/dummy_it"
		def __init__(self):
			self.LoadCheckpoint = MagicMock(name="LoadCheckpoint")
			self.SkipLoRA = MagicMock(name="SkipLoRA")
			self.CheckpointPath = _CKPTS.CheckpointPath

	gm = SimpleNamespace(ckpts=_CKPTS())

	# Inject gemma module so "from gemma import gm" inside functions uses this stub
	gemma_mod = types.ModuleType("gemma")
	gemma_mod.gm = gm
	monkeypatch.setitem(sys.modules, "gemma", gemma_mod)

	return gm


def test_create_standard_checkpoint_uses_variant_suffix(stub_gemma):
	out = Checkpoint.create_standard_checkpoint("DummyModel")
	stub_gemma.ckpts.LoadCheckpoint.assert_called_once_with(path="/ckpts/dummy_it")
	assert out is stub_gemma.ckpts.LoadCheckpoint.return_value


def test_create_lora_checkpoint_wraps_base(stub_gemma, monkeypatch):
	base = object()
	monkeypatch.setattr(ck_mod.Checkpoint, "create_standard_checkpoint", lambda *_: base)

	out = Checkpoint.create_lora_checkpoint("DummyModel")

	stub_gemma.ckpts.SkipLoRA.assert_called_once_with(wrapped=base)
	assert out is stub_gemma.ckpts.SkipLoRA.return_value