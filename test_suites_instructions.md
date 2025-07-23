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
   - For dialog/confirmation logic, test both “confirm” and “cancel” paths.

## 5. Running Tests

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

## 6. Next Steps

- Continue this process for each view/module.
- For backend, use `pytest` and mock external services as needed.
- Add integration/e2e tests after unit tests are stable.

---

**To resume:**  
Pick the next file, read it, summarize, decide what to test, and write the tests as above.