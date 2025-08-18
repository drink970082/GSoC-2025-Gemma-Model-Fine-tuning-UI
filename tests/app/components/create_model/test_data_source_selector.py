from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from tests.app.utils import setup_di_mock


def test_data_source_radio_buttons(monkeypatch):
    """Test data source radio buttons are displayed."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    data_source_radio = at.radio(key="data_source_selector")
    assert data_source_radio is not None
    assert data_source_radio.value in [
        "HuggingFace Dataset",
        "TensorFlow Dataset",
        "Custom JSON Upload",
    ]


def test_huggingface_source_selection(monkeypatch):
    """Test HuggingFace dataset source configuration."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    data_source_radio = at.radio(key="data_source_selector")
    data_source_radio.set_value("HuggingFace Dataset")
    at.run()

    dataset_name_input = at.text_input(key="dataset_name_input")
    assert dataset_name_input is not None
    assert (
        "google/fleurs, open-r1/Mixture-of-Thoughts"
        in dataset_name_input.placeholder
    )
    dataset_config_input = at.text_input(key="dataset_config_input")
    assert dataset_config_input is not None
    assert "hi_in, code" in dataset_config_input.placeholder
    assert (
        "Optional: Specify dataset-specific config" in dataset_config_input.help
    )
    split_input = at.text_input(key="split_input")
    assert split_input is not None
    assert "train, train[:80%], train[80%:]" in split_input.placeholder
    assert "Optional: Specify dataset split" in split_input.help


def test_tensorflow_source_selection(monkeypatch):
    """Test TensorFlow dataset source configuration."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    data_source_radio = at.radio(key="data_source_selector")
    data_source_radio.set_value("TensorFlow Dataset")
    at.run()

    dataset_name_input = at.text_input(key="dataset_name_input")
    assert dataset_name_input is not None
    assert "mtnt, mtnt/en-fr" in dataset_name_input.placeholder

    split_input = at.text_input(key="split_input")
    assert split_input is not None
    assert "Optional: Specify dataset split" in split_input.help


@patch("streamlit.file_uploader")
def test_json_upload_source_selection(mock_file_uploader, monkeypatch):
    """Test Custom JSON upload source configuration."""
    setup_di_mock(monkeypatch, MagicMock())

    mock_file_uploader.return_value = None

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    data_source_radio = at.radio(key="data_source_selector")
    data_source_radio.set_value("Custom JSON Upload")
    at.run()

    mock_file_uploader.assert_called_with(
        "Upload JSON file",
        type=["json"],
        help="Upload a JSON file containing your training data",
        key="uploaded_file_input",
    )


@patch("streamlit.file_uploader")
def test_json_upload_with_file(mock_file_uploader, monkeypatch):
    """Test JSON upload when file is provided."""
    setup_di_mock(monkeypatch, MagicMock())

    mock_file = MagicMock()
    mock_file.getvalue.return_value = b'{"test": "data"}'
    mock_file_uploader.return_value = mock_file

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    data_source_radio = at.radio(key="data_source_selector")
    data_source_radio.set_value("Custom JSON Upload")
    at.run()
    mock_file_uploader.assert_called()


def test_common_config_shuffle_checkbox(monkeypatch):
    """Test shuffle dataset checkbox."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    shuffle_checkbox = at.checkbox(key="shuffle_checkbox")
    assert shuffle_checkbox is not None
    assert shuffle_checkbox.value is True
    assert (
        "Whether to shuffle the dataset before training"
        in shuffle_checkbox.help
    )


def test_common_config_batch_size_slider(monkeypatch):
    """Test batch size slider."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    batch_size_slider = at.slider(key="batch_size_slider")
    assert batch_size_slider is not None
    assert batch_size_slider.min == 1
    assert batch_size_slider.max == 32
    assert batch_size_slider.value == 4
    assert (
        "Select the number of samples to process in each batch"
        in batch_size_slider.help
    )


def test_seq2seq_config_expander(monkeypatch):
    """Test sequence-to-sequence configuration expander."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    expanders = [
        exp
        for exp in at.expander
        if "Configure Sequence-to-Sequence Parameters" in exp.label
    ]
    assert len(expanders) > 0


def test_seq2seq_prompt_field_input(monkeypatch):
    """Test source field name input."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    prompt_field_input = at.text_input(key="prompt_field_input")
    assert prompt_field_input is not None
    assert (
        "Field name for source text (prompt) in the dataset"
        in prompt_field_input.help
    )


def test_seq2seq_response_field_input(monkeypatch):
    """Test target field name input."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    response_field_input = at.text_input(key="response_field_input")
    assert response_field_input is not None
    assert (
        "Field name for target text (response) in the dataset"
        in response_field_input.help
    )


def test_seq2seq_max_length_input(monkeypatch):
    """Test maximum sequence length input."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    max_length_input = at.number_input(key="max_length_input")
    assert max_length_input is not None
    assert max_length_input.min == 1
    assert max_length_input.max == 512
    assert "Maximum length of input sequences" in max_length_input.help


def test_seq2seq_truncate_checkbox(monkeypatch):
    """Test truncate sequences checkbox."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    truncate_checkbox = at.checkbox(key="truncate_checkbox")
    assert truncate_checkbox is not None
    assert truncate_checkbox.value is True
    assert (
        "Whether to truncate sequences longer than max_length"
        in truncate_checkbox.help
    )


def test_preview_dataset_button(monkeypatch):
    """Test preview dataset button."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    preview_button = at.button(key="preview_dataset")
    assert preview_button is not None
    assert preview_button.label == "Preview Dataset"


def test_data_source_change_updates_inputs(monkeypatch):
    """Test that changing data source updates the displayed inputs."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    # default is huggingface
    assert "dataset_config_input" in [inputs.key for inputs in at.text_input]

    data_source_radio = at.radio(key="data_source_selector")
    data_source_radio.set_value("TensorFlow Dataset").run()

    assert "dataset_config_input" not in [
        inputs.key for inputs in at.text_input
    ]


@patch("streamlit.file_uploader")
def test_json_upload_file_validation(mock_file_uploader, monkeypatch):
    """Test JSON file upload validation."""
    setup_di_mock(monkeypatch, MagicMock())

    mock_file_uploader.return_value = None

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    data_source_radio = at.radio(key="data_source_selector")
    data_source_radio.set_value("Custom JSON Upload")
    at.run()

    mock_file_uploader.assert_called_with(
        "Upload JSON file",
        type=["json"],
        help="Upload a JSON file containing your training data",
        key="uploaded_file_input",
    )


def test_default_values_persistence(monkeypatch):
    """Test that default values persist when switching between sources."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    shuffle_checkbox = at.checkbox(key="shuffle_checkbox")
    assert shuffle_checkbox.value is True

    batch_size_slider = at.slider(key="batch_size_slider")
    assert batch_size_slider.value == 4

    max_length_input = at.number_input(key="max_length_input")
    assert max_length_input.value == 200

    truncate_checkbox = at.checkbox(key="truncate_checkbox")
    assert truncate_checkbox.value is True


def test_help_text_display(monkeypatch):
    """Test that help text is displayed for all inputs."""
    setup_di_mock(monkeypatch, MagicMock())

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    inputs_with_help = [
        ("shuffle_checkbox", "Whether to shuffle the dataset before training"),
        (
            "batch_size_slider",
            "Select the number of samples to process in each batch",
        ),
        (
            "prompt_field_input",
            "Field name for source text (prompt) in the dataset",
        ),
        (
            "response_field_input",
            "Field name for target text (response) in the dataset",
        ),
        ("max_length_input", "Maximum length of input sequences"),
        (
            "truncate_checkbox",
            "Whether to truncate sequences longer than max_length",
        ),
    ]

    for key, expected_help in inputs_with_help:
        if key == "shuffle_checkbox":
            element = at.checkbox(key=key)
        elif key == "batch_size_slider":
            element = at.slider(key=key)
        elif key == "max_length_input":
            element = at.number_input(key=key)
        elif key == "truncate_checkbox":
            element = at.checkbox(key=key)
        else:
            element = at.text_input(key=key)

        assert element is not None
        assert expected_help in element.help


@patch("streamlit.file_uploader")
def test_json_upload_error_handling(mock_file_uploader, monkeypatch):
    """Test JSON upload error handling."""
    setup_di_mock(monkeypatch, MagicMock())

    mock_file = MagicMock()
    mock_file.getvalue.return_value = b"invalid json content"
    mock_file_uploader.return_value = mock_file

    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()

    data_source_radio = at.radio(key="data_source_selector")
    data_source_radio.set_value("Custom JSON Upload")
    at.run()

    mock_file_uploader.assert_called()
