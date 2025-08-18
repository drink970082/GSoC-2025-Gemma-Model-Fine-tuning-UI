# tests/backend/core/test_model.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import backend.core.model as model_mod
from backend.core.model import Model


@pytest.fixture(autouse=True)
def stub_gemma(monkeypatch):
    # gm stub
    class _NN:
        def __init__(self):
            self.called = {}
        def __getattr__(self, name):
            fn = MagicMock(name=f"gm.nn.{name}")
            self.called[name] = fn
            return fn
    class _CKPTS:
        def __init__(self):
            self.load_params = MagicMock(name="gm.ckpts.load_params")
    gm = SimpleNamespace(nn=_NN(), ckpts=_CKPTS())
    monkeypatch.setattr(model_mod, "gm", gm)
    return gm


@pytest.fixture(autouse=True)
def stub_peft(monkeypatch):
    class _QuantMethod:
        INT8 = "INT8"
    peft = SimpleNamespace(
        QuantizationMethod=_QuantMethod,
        quantize=MagicMock(name="peft.quantize"),
    )
    monkeypatch.setattr(model_mod, "peft", peft)
    return peft


def test_create_standard_model_calls_variant(stub_gemma):
    # Use a fake variant
    variant = "SomeVariant"
    result = Model.create_standard_model(variant)
    # Should call gm.nn.SomeVariant(tokens="batch.input")
    fn = stub_gemma.nn.called[variant]
    fn.assert_called_once_with(tokens="batch.input")
    assert result is fn.return_value


def test_create_lora_model_wraps_base(stub_gemma, monkeypatch):
    variant = "V"
    # Spy on base
    base = object()
    monkeypatch.setattr(Model, "create_standard_model", MagicMock(return_value=base))
    lora = MagicMock(name="LoRA")
    stub_gemma.nn.LoRA = lora

    out = Model.create_lora_model(variant, lora_rank=8)

    Model.create_standard_model.assert_called_once_with(variant)
    lora.assert_called_once_with(rank=8, model=base)
    assert out is lora.return_value


def test_create_quantization_aware_model_uses_wrapper(stub_gemma, stub_peft, monkeypatch):
    variant = "V"
    base = object()
    monkeypatch.setattr(Model, "create_standard_model", MagicMock(return_value=base))
    wrapper = MagicMock(name="QuantizationAwareWrapper")
    stub_gemma.nn.QuantizationAwareWrapper = wrapper

    out = Model.create_quantization_aware_model(variant)

    wrapper.assert_called_once_with(method=stub_peft.QuantizationMethod.INT8, model=base)
    assert out is wrapper.return_value


def test_create_quantization_aware_model_inference_uses_intwrapper(stub_gemma, monkeypatch):
    variant = "V"
    base = object()
    monkeypatch.setattr(Model, "create_standard_model", MagicMock(return_value=base))
    iw = MagicMock(name="IntWrapper")
    stub_gemma.nn.IntWrapper = iw

    out = Model.create_quantization_aware_model_inference(variant)

    iw.assert_called_once_with(model=base)
    assert out is iw.return_value


def test_load_trained_params_standard(stub_gemma):
    params = object()
    stub_gemma.ckpts.load_params.return_value = params

    out = Model.load_trained_params("/ckpt", method="Standard")
    stub_gemma.ckpts.load_params.assert_called_once_with("/ckpt")
    assert out is params


def test_load_trained_params_quantization_aware(stub_gemma, stub_peft):
    params = object()
    qparams = object()
    stub_gemma.ckpts.load_params.return_value = params
    stub_peft.quantize.return_value = qparams

    out = Model.load_trained_params("/ckpt", method="QuantizationAware")

    stub_gemma.ckpts.load_params.assert_called_once_with("/ckpt")
    stub_peft.quantize.assert_called_once_with(params, method=stub_peft.QuantizationMethod.INT8)
    assert out is qparams