# Test Suite Building Process

## 1. Directory Structure

- All tests live under a top-level `tests/` directory.
- Mirror the app structure for clarity:
  - `tests/app/test_main.py` (for `app/main.py`)
  - `tests/app/view/test_welcome_view.py` (for `app/view/welcome_view.py`)
  - etc.

## 2. Test Frameworks

- **Backend:** Use `pytest`
- **Streamlit UI:** Use `streamlit.testing.v1` (`AppTest`)

## 3. Test Types & Priorities

- Start with **unit/UI tests** for each module/view.
- Add **integration** and **e2e** tests after unit coverage is solid.

## 4. Test Writing Workflow

For each file:
1. **Read the file.**
2. **Summarize main components** (functions/classes/logic).
3. **Decide what to test:**  
   - Focus on user-facing logic, state changes, and custom code.
   - Skip trivial wiring/boilerplate and Streamlit internals.
4. **Write tests:**
   - Use `AppTest` for Streamlit views.
   - Mock slow or external dependencies (e.g., backend services).
   - Simulate button clicks and check `st.session_state` and UI output.
   - For dialog/confirmation logic, test both "confirm" and "cancel" paths.

## 5. Streamlit Testing Best Practices

### **DO: Use AppTest.from_file() with DI mocking**
```python
def _setup_di_mock(monkeypatch, training_service):
    """Helper to setup DI container mocking."""
    def mock_get_service(name):
        if name == "training_service":
            return training_service
        return MagicMock()
    
    monkeypatch.setattr("services.di_container.get_service", mock_get_service)

def test_some_view(monkeypatch, mock_training_service):
    _setup_di_mock(monkeypatch, mock_training_service)
    
    at = AppTest.from_file("app/main.py")
    at.session_state["view"] = "target_view"  # Set the view you want to test
    at.run()
    
    # Test assertions here
    assert "expected text" in [info.value for info in at.info]
```

### **DON'T: Use AppTest.from_function() with closures/lambdas**
```python
# This WILL FAIL - closures don't work with AppTest.from_function()
def test_broken_approach(mock_service):
    def app():
        from app.view.some_view import show_view
        show_view(mock_service)  # NameError: mock_service not defined
    
    at = AppTest.from_function(app)  # Serialization breaks closure
```

### **Key Patterns:**
- **Always use `AppTest.from_file("app/main.py")`** - provides real app context
- **Mock at DI container level** using `monkeypatch.setattr("services.di_container.get_service", mock_func)`
- **Control views via session state** - set `at.session_state["view"]` before running
- **Create helper fixtures** for different service states (e.g., RUNNING vs IDLE)
- **Use list comprehensions** for element searches: `[info.value for info in at.info]`
- **Call `at.run()` after state changes** to re-render the UI

## 6. Running Tests

From project root:
```bash
pytest
```
- To run a specific test file:
  ```bash
  pytest tests/app/view/test_welcome_view.py
  ```
- For verbose output:
  ```bash
  pytest -v
  ```

## 7. Next Steps

- Continue this process for each view/module.
- For backend, use `pytest` and mock external services as needed.
- Add integration/e2e tests after unit tests are stable.

---

**To resume:**  
Pick the next file, read it, summarize, decide what to test, and write the tests using the **AppTest.from_file() + DI mocking** pattern above.