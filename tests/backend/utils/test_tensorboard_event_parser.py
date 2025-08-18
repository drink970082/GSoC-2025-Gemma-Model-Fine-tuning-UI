# tests/backend/utils/test_tensorboard_event_parser.py
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import backend.utils.tensorboard_event_parser as tp_mod
from backend.utils.tensorboard_event_parser import (
    EventFileParser,
    DT_FLOAT,
    DT_INT32,
    DT_INT64,
    DT_STRING,
    EVENT_FILE_PATTERN,
)


@dataclass
class _TensorProto:
    dtype: int
    tensor_shape: Any
    float_val: list[float] | None = None
    int64_val: list[int] | None = None
    string_val: list[bytes] | None = None
    tensor_content: bytes | None = None


@dataclass
class _Event:
    wall_time: float
    step: int
    tensor_proto: _TensorProto


class _FakeEA:
    def __init__(self, path, size_guidance=None):
        self.path = path
        self._tags = {"tensors": []}
        self._tensors = {}

    def Reload(self):
        return None

    def Tags(self):
        return self._tags

    def Tensors(self, tag):
        return self._tensors.get(tag, [])

    # Helpers to seed
    def seed(self, tensors_map):
        self._tags = {"tensors": list(tensors_map.keys())}
        self._tensors = tensors_map


@pytest.fixture
def fake_ea(monkeypatch):
    inst = {"ea": None}

    def ctor(path, size_guidance=None):
        ea = _FakeEA(path, size_guidance)
        inst["ea"] = ea
        return ea

    monkeypatch.setattr(tp_mod, "EventAccumulator", ctor)
    return inst


def test_find_event_file_none(tmp_path):
    p = EventFileParser(str(tmp_path))
    assert p._find_event_file() is None


def test_find_event_file_found(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / f"{EVENT_FILE_PATTERN}.123").write_text("x", encoding="utf-8")
    p = EventFileParser(str(tmp_path))
    found = p._find_event_file()
    assert found is not None
    assert EVENT_FILE_PATTERN in os.path.basename(found)

def test_load_event_data_end_to_end(tmp_path, monkeypatch):
    # Arrange parser to find a file
    p = EventFileParser(str(tmp_path))
    monkeypatch.setattr(p, "_find_event_file", lambda: str(tmp_path / "events.mock"))

    # Build tensor protos
    params_text = "Total: 1,234 -- 9,876 bytes\n| layer_0/foo | (1,) | bfloat16 | 1 |"
    params_event = _Event(
        wall_time=1.0,
        step=0,
        tensor_proto=_TensorProto(
            dtype=DT_STRING, tensor_shape=SimpleNamespace(dim=[1]), string_val=[params_text.encode("utf-8")]
        ),
    )
    loss_events = [
        _Event(1.0, 1, _TensorProto(dtype=DT_FLOAT, tensor_shape=SimpleNamespace(dim=[1]), float_val=[0.5])),
        _Event(2.0, 2, _TensorProto(dtype=DT_FLOAT, tensor_shape=SimpleNamespace(dim=[1]), float_val=[0.4])),
    ]
    arr = np.array([10], dtype=np.int64).tobytes()
    steps_event = _Event(
        1.5,
        2,
        _TensorProto(dtype=DT_INT64, tensor_shape=SimpleNamespace(dim=[1]), tensor_content=arr),
    )
    other = _Event(1.0, 1, _TensorProto(dtype=DT_INT32, tensor_shape=SimpleNamespace(dim=[1]), int64_val=None))

    tensors_map = {
        "parameters": [params_event],
        "losses/loss": loss_events,
        "perf_stats/steps_per_sec": [steps_event],
        "other/tag": [other],
    }

    # Ensure the instance constructed inside load_event_data is the one we seed
    def ctor(path, size_guidance=None):
        ea = _FakeEA(path, size_guidance)
        ea.seed(tensors_map)
        return ea

    monkeypatch.setattr(tp_mod, "EventAccumulator", ctor)

    # Act
    out = p.load_event_data()

    # Assert metadata parsed
    md = out["metadata"]
    assert md["parameters"]["total_params"] == 1234
    assert md["parameters"]["total_bytes"] == 9876
    assert len(md["parameters"]["layers"]) >= 1

    # Training metrics DataFrames present
    tms = out["training_metrics"]
    assert "losses/loss" in tms and "perf_stats/steps_per_sec" in tms
    assert isinstance(tms["losses/loss"], pd.DataFrame)

    # Latest metrics
    latest = out["latest_training_metrics"]
    assert latest["losses/loss"] == 0.4
    assert latest["perf_stats/steps_per_sec"] == 10.0


def test_parse_tensor_value_matrix():
    # float_val
    tp = _TensorProto(dtype=DT_FLOAT, tensor_shape=SimpleNamespace(dim=[1]), float_val=[3.14])
    assert EventFileParser("x")._parse_tensor_value(tp) == 3.14

    # int64_val short-circuit
    tp = _TensorProto(dtype=DT_INT64, tensor_shape=SimpleNamespace(dim=[1]), int64_val=[7])
    assert EventFileParser("x")._parse_tensor_value(tp) == 7

    # string_val
    tp = _TensorProto(dtype=DT_STRING, tensor_shape=SimpleNamespace(dim=[1]), string_val=[b"hello"])
    assert EventFileParser("x")._parse_tensor_value(tp) == "hello"

    # tensor_content float
    arr = np.array([1.25], dtype=np.float32).tobytes()
    tp = _TensorProto(dtype=DT_FLOAT, tensor_shape=SimpleNamespace(dim=[1]), tensor_content=arr)
    assert EventFileParser("x")._parse_tensor_value(tp) == 1.25

    # tensor_content int32
    arr = np.array([42], dtype=np.int32).tobytes()
    tp = _TensorProto(dtype=DT_INT32, tensor_shape=SimpleNamespace(dim=[1]), tensor_content=arr)
    assert EventFileParser("x")._parse_tensor_value(tp) == 42

    # tensor_content int64
    arr = np.array([99], dtype=np.int64).tobytes()
    tp = _TensorProto(dtype=DT_INT64, tensor_shape=SimpleNamespace(dim=[1]), tensor_content=arr)
    assert EventFileParser("x")._parse_tensor_value(tp) == 99

    # tensor_content string (non-utf8)
    tp = _TensorProto(dtype=DT_STRING, tensor_shape=SimpleNamespace(dim=[1]), tensor_content=b"\xff\xfe")
    v = EventFileParser("x")._parse_tensor_value(tp)
    assert isinstance(v, str) and "String tensor" in v

    # default fallback string
    tp = _TensorProto(dtype=999, tensor_shape=SimpleNamespace(dim=[1, 2]), tensor_content=b"")
    v = EventFileParser("x")._parse_tensor_value(tp)
    assert v.startswith("Shape:") and "Dtype:" in v


def test_parse_parameter_summary():
    text = """
    | header |
    | layer_0/foo | (1,) | bfloat16 | 1 |
    Total: 1,234 -- 9,876 bytes
    """
    p = EventFileParser("x")
    out = p._parse_parameter_summary(text)
    assert out["total_params"] == 1234
    assert out["total_bytes"] == 9876
    assert out["parameter_count"] >= 1
    assert any("layer_0" in l["name"] for l in out["layers"])


def test_parse_element_spec():
    text = 'junk {"input": {"shape": [4, 200], "dtype": "i32"}, "loss_mask": {"shape": [4, 200]}, "target": {"shape": [4, 200]}} trailer'
    p = EventFileParser("x")
    out = p._parse_element_spec(text)
    assert out["input_shape"] == [4, 200]
    assert out["input_dtype"] == "i32"
    assert out["loss_mask_shape"] == [4, 200]
    assert out["target_shape"] == [4, 200]
    assert out["batch_size"] == 4
    assert out["sequence_length"] == 200


def test_parse_context_spec():
    text = """
    | Path | Spec |
    | `batch.input` | `i32[4 200]` |
    | `grads.foo` | `bf16[4 200]` |
    """
    p = EventFileParser("x")
    out = p._parse_context_spec(text)
    assert out["total_specs"] == 2
    assert any(s["path"] == "batch.input" for s in out["batch_specs"])
    assert any(s["path"] == "grads.foo" for s in out["grad_specs"])