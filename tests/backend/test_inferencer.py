import json
import os
import shutil
from unittest.mock import MagicMock, mock_open, patch

import pytest

from backend.inferencer import MODEL_CONFIG_FILE, Inferencer

# --- Checkpoint Management ---


def test_list_checkpoints_empty(tmp_path):
    inf = Inferencer(work_dir=str(tmp_path))
    assert inf.list_checkpoints() == []


def test_list_checkpoints_sorted(tmp_path):
    # Create two checkpoint dirs with different ctimes
    d1 = tmp_path / "ckpt1"
    d2 = tmp_path / "ckpt2"
    d1.mkdir()
    d2.mkdir()
    os.utime(d1, (1, 1))
    os.utime(d2, (2, 2))
    inf = Inferencer(work_dir=str(tmp_path))
    # Should be sorted newest first
    assert inf.list_checkpoints() == ["ckpt2", "ckpt1"]


def test_get_latest_checkpoint(tmp_path):
    d1 = tmp_path / "ckpt1"
    d2 = tmp_path / "ckpt2"
    d1.mkdir()
    d2.mkdir()
    os.utime(d1, (1, 1))
    os.utime(d2, (2, 2))
    inf = Inferencer(work_dir=str(tmp_path))
    assert inf.get_latest_checkpoint() == "ckpt2"


def test_delete_checkpoint(tmp_path):
    d1 = tmp_path / "ckpt1"
    d1.mkdir()
    inf = Inferencer(work_dir=str(tmp_path))
    assert inf.delete_checkpoint("ckpt1") is True
    assert not d1.exists()
    assert inf.delete_checkpoint("ckpt1") is False


# --- Model Loading ---


@patch("backend.inferencer.Model")
def test_load_model_success(mock_model, tmp_path):
    # Setup checkpoint dir and config file
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    config = {
        "model_variant": "foo",
        "epochs": 1,
        "learning_rate": 0.1,
        "method": "Standard",
    }
    (ckpt_dir / MODEL_CONFIG_FILE).write_text(json.dumps(config))

    # Patch _get_checkpoint_path to return ckpt_dir
    inf = Inferencer(work_dir=str(tmp_path))
    inf._get_checkpoint_path = lambda wd: str(ckpt_dir)

    # Patch model creation
    mock_model.create_standard_model.return_value = "model"
    mock_model.load_trained_params.return_value = "params"

    # Patch tokenizer/sampler at the import location
    with patch("gemma.gm.text.Gemma3Tokenizer") as mock_tok, patch(
        "gemma.gm.text.ChatSampler"
    ) as mock_sampler:
        mock_tok.return_value = MagicMock()
        mock_sampler.return_value = MagicMock(chat=lambda prompt: "output")

        assert inf.load_model(checkpoint_path="ckpt") is True
        assert inf.is_loaded()
        assert inf.model == "model"
        assert inf.params == "params"
        assert inf.tokenizer is not None
        assert inf.sampler is not None


def test_load_model_no_checkpoint(tmp_path):
    inf = Inferencer(work_dir=str(tmp_path))
    assert inf.load_model() is False


@patch("os.path.exists", return_value=False)
def test_load_model_checkpoint_missing(mock_exists, tmp_path):
    inf = Inferencer(work_dir=str(tmp_path))
    assert inf.load_model(checkpoint_path="ckpt") is False


@patch("backend.inferencer.open", new_callable=mock_open, read_data="bad json")
@patch("os.path.exists", return_value=True)
def test_load_model_bad_config(mock_exists, mock_openfile, tmp_path):
    inf = Inferencer(work_dir=str(tmp_path))
    # Patch _get_checkpoint_path to return a valid path
    inf._get_checkpoint_path = lambda wd: str(tmp_path)
    assert inf.load_model(checkpoint_path="ckpt") is False


# --- Generation and Token Counting ---


def test_generate_and_count_tokens(monkeypatch):
    inf = Inferencer()
    inf._loaded = True
    inf.sampler = MagicMock(chat=lambda prompt: "generated text")
    inf.tokenizer = MagicMock(encode=lambda text: [1, 2, 3])
    assert inf.generate("prompt") == "generated text"
    assert inf.count_tokens("foo bar") == 3


def test_generate_not_loaded():
    inf = Inferencer()
    with pytest.raises(RuntimeError):
        inf.generate("prompt")


def test_count_tokens_not_loaded():
    inf = Inferencer()
    assert inf.count_tokens("foo") == 0


# --- State Management ---


def test_clear_model():
    inf = Inferencer()
    inf.model = "model"
    inf.params = "params"
    inf.tokenizer = "tokenizer"
    inf.sampler = "sampler"
    inf._loaded = True
    inf.clear_model()
    assert inf.model is None
    assert inf.params is None
    assert inf.tokenizer is None
    assert inf.sampler is None
    assert not inf._loaded


# --- Internal Helpers ---


def test_parse_model_config_standard():
    inf = Inferencer()
    d = {
        "model_variant": "foo",
        "epochs": 1,
        "learning_rate": 0.1,
        "method": "Standard",
    }
    cfg = inf._parse_model_config(d)
    assert cfg.model_variant == "foo"
    assert cfg.method == "Standard"


def test_parse_model_config_lora():
    inf = Inferencer()
    d = {
        "model_variant": "foo",
        "epochs": 1,
        "learning_rate": 0.1,
        "method": "LoRA",
        "parameters": {"lora_rank": 8},
    }
    cfg = inf._parse_model_config(d)
    assert cfg.method == "LoRA"
    assert cfg.parameters.lora_rank == 8


@patch("backend.inferencer.Model")
def test_create_model_from_config_standard(mock_model):
    inf = Inferencer()
    cfg = MagicMock(method="Standard", model_variant="foo")
    inf._create_model_from_config(cfg)
    mock_model.create_standard_model.assert_called_with("foo")


@patch("backend.inferencer.Model")
def test_create_model_from_config_lora(mock_model):
    inf = Inferencer()
    params = MagicMock(lora_rank=8)
    cfg = MagicMock(method="LoRA", model_variant="foo", parameters=params)
    inf._create_model_from_config(cfg)
    mock_model.create_lora_model.assert_called_with("foo", 8)


@patch("backend.inferencer.Model")
def test_create_model_from_config_quant(mock_model):
    inf = Inferencer()
    cfg = MagicMock(method="QuantizationAware", model_variant="foo")
    inf._create_model_from_config(cfg)
    mock_model.create_quantization_aware_model_inference.assert_called_with(
        "foo"
    )
