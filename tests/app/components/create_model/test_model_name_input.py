from streamlit.testing.v1 import AppTest


def test_model_name_input_placeholder():
    """Test that model name input shows correct placeholder."""
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    assert (
        at.text_input(key="model_name").placeholder == "e.g., gemma-3-1b-LoRA"
    )


def test_model_name_input_help_text():
    """Test that model name input shows help text."""
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    assert (
        "Choose a descriptive name for your fine-tuned model"
        in at.text_input(key="model_name").help
    )
