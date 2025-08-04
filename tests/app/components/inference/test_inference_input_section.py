from unittest.mock import MagicMock

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.components.inference.utils import mock_inferencer
from tests.app.utils import setup_di_mock


def test_shows_inference_input_section(monkeypatch, mock_inferencer):
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()
    assert at.text_area(key="prompt_input").label == "Enter your prompt:"
    assert any(btn.label == "Generate Response" for btn in at.button)


def test_generate_response_with_valid_prompt(monkeypatch, mock_inferencer):
    setup_di_mock(monkeypatch, MagicMock())
    mock_inferencer.count_tokens.return_value = 10

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()
    at.text_area(key="prompt_input").set_value("Test prompt")
    at.run()
    at.button(key="generate_response").click()
    at.run()
    mock_inferencer.generate.assert_called_once_with("Test prompt")
    print(at.subheader)
    assert "Response" in [sub.body for sub in at.subheader]
    assert "Input tokens: 10 | Output tokens: 10" in [
        caption.body for caption in at.caption
    ]


def test_generate_response_with_empty_prompt(monkeypatch, mock_inferencer):
    setup_di_mock(monkeypatch, MagicMock())
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()
    at.button(key="generate_response").click()
    at.run()

    assert "Please enter a prompt." in [warning.value for warning in at.warning]


def test_generate_response_no_model_loaded(monkeypatch, mock_inferencer):
    mock_inferencer.is_loaded.return_value = False
    setup_di_mock(monkeypatch, MagicMock())
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()
    at.text_area(key="prompt_input").set_value("Test prompt")
    at.run()
    at.button(key="generate_response").click()
    at.run()

    assert "No model loaded. Please load a checkpoint first." in [
        error.value for error in at.error
    ]


def test_generate_response_exception_handling(monkeypatch, mock_inferencer):
    mock_inferencer.generate.side_effect = Exception("Generation failed")
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "inference"
    at.run()
    at.text_area(key="prompt_input").set_value("Test prompt")
    at.run()
    at.button(key="generate_response").click()
    at.run()

    assert "Error during generation: Generation failed" in [
        error.value for error in at.error
    ]
