import streamlit as st
from streamlit.testing.v1 import AppTest


def test_app_loads():
    at = AppTest.from_file("app/main.py")
    at.run()
    assert at.title[0].value == "Gemma Fine-Tuning"


def test_sidebar_navigation_sets_welcome(monkeypatch):
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "create_model"
    at.run()
    at.button(key="back_to_home").click()
    at.run()
    assert at.session_state["view"] == "welcome"


def test_view_switching():
    at = AppTest.from_file("app/main.py")
    header = {
        "welcome": "Gemma Fine-Tuning",
        "create_model": "Create Model",
        "training_dashboard": "LLM Fine-Tuning Dashboard",
        "inference": "Inference Playground",
    }
    for view in ["welcome", "create_model", "training_dashboard", "inference"]:
        at.session_state["view"] = view
        at.run()
        assert at.title[0].value == header[view]
