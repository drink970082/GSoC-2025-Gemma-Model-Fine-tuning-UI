# Backend Test Suite Plan

## 1. Directory Structure

All backend tests should live under `tests/backend/`, mirroring the backend source tree:
```bash
tests/backend/
├── test_inferencer.py
├── test_data_pipeline.py
├── test_trainer_main.py
├── manager/
│ ├── test_tensorboard_manager.py
│ ├── test_process_manager.py
│ └── ...
├── core/
│ ├── test_model_loader.py
│ ├── test_tokenizer.py
│ └── ...
├── utils/
│ ├── test_file_utils.py
│ └── ...
└── init.py
```

## 2. Test Types

- **Unit tests** for all functions/classes
- **Integration tests** for cross-module flows (e.g., trainer + pipeline)
- **Mocking** for external dependencies (file system, subprocess, etc.)
- **Edge case and error handling** for all public APIs

## 3. Task Breakdown

### A. Top-Level Files

#### 1. `inferencer.py`
- Test all public methods (inference, batch inference, error handling)
- Mock model loading, tokenizer, and data input
- Test output formatting and exception paths

#### 2. `data_pipeline.py`
- Test data loading, preprocessing, batching, and augmentation
- Mock file I/O and external data sources
- Test edge cases (empty data, corrupt data, etc.)

#### 3. `trainer_main.py`
- Test CLI entry points, argument parsing, and main training loop
- Mock model, optimizer, and data pipeline
- Test error handling and logging

### B. Subdirectories

#### 4. `manager/`
- Test each manager (TensorBoard, Process, etc.)
- Mock subprocesses, file I/O, and system calls
- Test state transitions, error recovery, and resource cleanup

#### 5. `core/`
- Test core abstractions (model loader, tokenizer, etc.)
- Mock model files, configs, and external libraries
- Test compatibility and error handling

#### 6. `utils/`
- Test all utility functions (file ops, config parsing, etc.)
- Mock file system and environment as needed
- Test edge cases and invalid input

## 4. Implementation Phases

### **Phase 1: Skeleton & Smoke Tests**
- Create empty test files for each module
- Add basic import/smoke tests to ensure all modules load

### **Phase 2: Unit Tests for Top-Level Files**
- Write detailed unit tests for `inferencer.py`, `data_pipeline.py`, `trainer_main.py`
- Use pytest fixtures and mocks for dependencies

### **Phase 3: Manager & Core Modules**
- Write unit tests for each manager and core module
- Focus on state transitions, error handling, and resource management

### **Phase 4: Utils & Integration**
- Write tests for utility functions
- Add integration tests for key flows (e.g., data pipeline + trainer)

### **Phase 5: Edge Cases & Error Handling**
- Add tests for invalid input, exceptions, and recovery logic

## 5. Test Writing Workflow

For each file:
1. **Read the file.**
2. **Summarize main components** (functions/classes/logic).
3. **Decide what to test:**  
   - Focus on public APIs, state changes, and custom logic.
   - Skip trivial wiring/boilerplate.
4. **Write tests:**
   - Use pytest and unittest.mock.
   - Mock slow/external dependencies.
   - Simulate error conditions and edge cases.
5. **Run and iterate.**

## 6. Example Test Skeleton

```python
# tests/backend/test_inferencer.py

import pytest
from backend.inferencer import Inferencer

def test_inferencer_basic(monkeypatch):
    # Mock model and tokenizer
    ...
    inf = Inferencer(...)
    result = inf.infer("test input")
    assert result == "expected output"
```

## 7. Next Steps

- Start with Phase 1: create the test skeletons and smoke tests for all modules.
- Proceed module by module, following the workflow above.
- Use the same thoroughness and mocking patterns as in the Streamlit/UI tests.

---