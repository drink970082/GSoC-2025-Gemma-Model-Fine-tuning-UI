from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest
from tests.app.utils import setup_di_mock


def test_model_selector_section(monkeypatch):
    """Test model selector section is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    model_selectbox = at.selectbox(key="gemma_model_selector")
    assert model_selectbox is not None
    assert model_selectbox.value in [
        "Gemma3_1B",
        "Gemma3_2B",
        "Gemma3_7B",
        "Gemma3_9B",
    ]
    assert (
        "Choose the model size based on your task and available resources"
        in model_selectbox.help
    )
    assert model_selectbox.value == "Gemma3_1B"


def test_fine_tuning_method_selection(monkeypatch):
    """Test fine-tuning method radio buttons are displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    method_radio = at.radio(key="fine_tuning_method_selector")
    assert method_radio is not None
    assert (
        "Choose the fine-tuning approach based on your needs"
        in method_radio.help
    )


def test_model_info_display(monkeypatch):
    """Test model information is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    assert any("Size:" in info.body for info in at.info)


def test_model_details_expander(monkeypatch):
    """Test model details expander is present."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    expanders = [exp for exp in at.expander if "Model Details" in exp.label]
    assert len(expanders) > 0


def test_method_info_display(monkeypatch):
    """Test method information is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    info_texts = [info.body for info in at.info]
    assert len(info_texts) >= 2


def test_method_details_expander(monkeypatch):
    """Test method details expander is present."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    expanders = [exp for exp in at.expander if "Method Details" in exp.label]
    assert len(expanders) > 0


def test_training_parameters_section(monkeypatch):
    """Test training parameters section is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    markdown_texts = [md.value for md in at.markdown]
    assert any("#### Training Parameters" in text for text in markdown_texts)


def test_epochs_input(monkeypatch):
    """Test epochs number input is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    epochs_input = at.number_input(key="epochs_input")
    assert epochs_input is not None
    assert epochs_input.min == 1
    assert "Enter the total number of training epochs" in epochs_input.help


def test_learning_rate_input(monkeypatch):
    """Test learning rate number input is displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    lr_input = at.number_input(key="learning_rate_input")
    assert lr_input is not None
    assert lr_input.min == 1e-6
    assert lr_input.max == 1e-2
    assert (
        "Set the learning rate. Higher values are faster but more unstable, lower values are slower but more stable."
        in lr_input.help
    )


def test_lora_parameters_when_lora_selected(monkeypatch):
    """Test LoRA parameters are shown when LoRA method is selected."""
    setup_di_mock(monkeypatch, MagicMock())
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    method_radio = at.radio(key="fine_tuning_method_selector")
    method_radio.set_value("LoRA")
    at.run()
    markdown_texts = [md.value for md in at.markdown]
    assert any("#### LoRA Parameters" in text for text in markdown_texts)
    lora_rank_input = at.number_input(key="lora_rank_input")
    assert lora_rank_input is not None
    assert lora_rank_input.min == 1
    assert lora_rank_input.max == 32
    assert lora_rank_input.help is not None


def test_standard_method_no_parameters(monkeypatch):
    """Test Standard method doesn't show additional parameters."""
    setup_di_mock(monkeypatch, MagicMock())
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    method_radio = at.radio(key="fine_tuning_method_selector")
    method_radio.set_value("Standard")
    at.run()
    markdown_texts = [md.value for md in at.markdown]
    assert not any("#### LoRA Parameters" in text for text in markdown_texts)


def test_model_selection_change(monkeypatch):
    """Test model selection changes update the display."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    model_selectbox = at.selectbox(key="gemma_model_selector")
    original_value = model_selectbox.value
    model_selectbox.set_value("Gemma3_4B")
    at.run()
    assert model_selectbox.value == "Gemma3_4B"
    assert model_selectbox.value != original_value


def test_method_selection_change(monkeypatch):
    """Test method selection changes update the display."""
    setup_di_mock(monkeypatch, MagicMock())
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    method_radio = at.radio(key="fine_tuning_method_selector")
    method_radio.set_value("LoRA")
    at.run()
    markdown_texts = [md.value for md in at.markdown]
    assert any("#### LoRA Parameters" in text for text in markdown_texts)
