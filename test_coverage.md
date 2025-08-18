# Test Coverage & Quality Report

## TL;DR
- Broad, meaningful coverage across `app/`, `backend/`, `services/`.
- Gaps: `backend/core/loss.py`, `backend/core/sampler.py`, `data preview` click-path, some private helpers, and `Inferencer` checkpoint discovery edge cases.
- Product risks: brittle checkpoint path logic; method/variant validation; noisy logs error counting; preview rendering error boundaries; relative trainer path.

---

## 1) Coverage: file-by-file

### app/
- `app/main.py`: Partial
  - Covered: view switching, sidebar nav.
  - Not covered: `_initialize_session_state()` keys; `default_config()` sanity.
- `app/view/welcome_view.py`: Good (running/abort/nav flows).
- `app/view/create_model_view.py`: Good (sections; start training success/failure; session flags).
- `app/view/training_dashboard_view.py`: Good (polling fragment transitions; sections).
- `app/view/inference_view.py`: Good (title/sections).

- `app/components/create_model/`
  - `model_name_input.py`: Covered.
  - `data_source_selector.py`: Partial
    - Covered: source UIs; help; radio switching; JSON upload (none/invalid).
    - Missing: “Preview Dataset” button path (raw/tokenized tabs, `treescope` rendering, decode-loop error handling).
  - `model_selector.py`: Covered (info panels; LoRA params; limits).
  - `start_training_button.py`: Good (all validations, pipeline happy/exception).
  - `config_summary.py`: Covered (valid/none/error).

- `app/components/training_dashboard/`
  - `control_panel.py`: Covered (RUNNING/FINISHED/FAILED/aborted; checkpoint finder).
  - `kpi_panel.py`: Good (waiting; metadata/training/perf; time formatting; frozen; fragment).
  - `plots_panel.py`: Good (waiting; loss/perf subsets; empty/single point; invalid df; frozen; fragment).
  - `logs_panel.py`: Good (stdout/stderr variants; frozen; error counting).
  - `system_usage_panel.py`: Covered (no GPU; not enough points; CPU-only; mixed; exceptions).

- `app/components/inference/`
  - `checkpoint_selection.py`: Covered (select/load/delete/no checkpoints).
  - `inference_input_section.py`: Covered (empty prompt; not loaded; happy/exception; token counts).

### backend/core
- `model.py`: Covered (Standard/LoRA/QAT factories; `load_trained_params()` for Standard/QAT).
- `checkpoint.py`: Covered (standard path + _IT; LoRA wrapper).
- `optimizer.py`: Covered (Adafactor; LoRA partial updates).
- `fine_tuner.py`: Covered (base kd.Trainer wiring; three strategies).
- `trainer.py`: Covered (env var; happy path; traceback + re-raise).
- `loss.py`: Not covered.
- `sampler.py`: Not covered.

### backend
- `data_pipeline.py`: Good (base helpers; JSON/TF/HF tokenized/preview; transforms; split; env; error handling).
- `inferencer.py`: Covered (list/sort/delete; happy `load_model`; not-loaded guard; parsing; model switch).
  - Missing: `_get_checkpoint_path()` edge cases (no dir/empty/ordering/type).
- `utils/cli.py`: Covered (json type + required args).
- `utils/tensorboard_event_parser.py`: Good (event find/load; tensor parsing; metadata/element/context; metrics aggregation; fallbacks).

### backend/manager
- `base_manager.py`: Covered.
- `system_manager.py`: Covered (NVML present/absent; shutdown; polling; errors; units).
- `process_manager.py`: Good (start; already running; early-exit failure; missing config/workdir; graceful/timeout/force; force_cleanup variants; dead-process success/failure; running-without-proc).
  - Missing: `_is_process_running()`; `_open_log_files()` OSError; `_handle_dead_process()` orphaned branch.
- `training_state_manager.py`: Covered (all state transitions incl atomicity).
- `tensorboard_manager.py`: Covered (KPI/metrics; ETA; exceptions; cleanup; workdir propagation).

### services
- `training_service.py`: Covered (running guard; start happy/fail; stop graceful/force; orphaned cleanup; FINISHED/FAILED resets; getters/logs; workdir lifecycle).
  - Missing: `_wait_for_state_file` timeout raising branch.
- `di_container.py`: Covered (one-time setup; wiring; atexit; cleanup exceptions).

### config
- `app_config.py`: Not covered (constants/singleton).
- `dataclass.py`: Used implicitly via other tests.
- `fine_tuning_info.py`, `model_info.py`: Not covered (reference data).

---

## 2) Gaps in tests (not covering real code paths)

- `backend/core/loss.py`: no test for `kd.losses.SoftmaxCrossEntropyWithIntLabels` wiring.
- `backend/core/sampler.py`: no test for `gm.text.ChatSampler` usage.
- `backend/inferencer.py`:
  - `_get_checkpoint_path()` untested: missing `checkpoints/` dir; empty dir; newest selection; return type consistency.
  - `load_model()` error branches where tokenizer/sampler creation fails.
- `backend/manager/process_manager.py`:
  - `_is_process_running()` (os.kill probe).
  - `_open_log_files()` OSError handling (closing partial handle).
- `services/training_service.py`:
  - `_wait_for_state_file` timeout raising path.
- `app/components/create_model/data_source_selector.py`:
  - Preview-button flow: raw preview, tokenized preview (`treescope`) + decode loop; error surfaces in those blocks.
- `app/main.py`:
  - `_initialize_session_state()` keys smoke test; `default_config()` validity check.
- `logs_panel`:
  - Current tests intentionally accept substring “error”; if you tighten matching, add tests accordingly.
- `config/*`: no smoke tests to ensure expected keys exist for UI.

---

## 3) Product issues (robustness/defects)

- Inferencer checkpoint path is brittle and returns `Path`:
  ```text
  backend/inferencer.py::_get_checkpoint_path()
  - Fails if <work_dir>/checkpoints/ missing or empty (iterdir/index errors).
  - Picks arbitrary first subdir (not newest).
  - Returns Path vs str.
  ```
- Unknown `model_config.method` in `ModelTrainer` → `KeyError` from `FINE_TUNE_STRATEGIES`.
  - Should raise `ValueError("Unknown fine-tune method: ...")`.
- `Model.create_standard_model()` invalid `model_variant` → `AttributeError`.
  - Consider validation against `config.app_config.MODEL_OPTIONS` for friendlier error.
- `logs_panel._count_errors()` counts any “error” substring (e.g., “not an error”).
  - Decide if this is desired; otherwise use word-boundary or structured prefixes.
- `system_usage_panel.py`: stray `print(available_charts)` on every render.
- `data_source_selector` preview: `treescope` rendering/decoding errors aren’t wrapped; only pipeline creation is guarded.
- `TRAINER_MAIN_PATH` is relative; subprocess depends on CWD. Use absolute path.
- Minor: `services/di_container.py` log typo “ATEIXT”.

---

## 4) Suggestions / next steps

### Tests to add (fast wins)
- `tests/backend/core/test_loss.py`:
  - Monkeypatch `kd.losses.SoftmaxCrossEntropyWithIntLabels`; assert args.
- `tests/backend/core/test_sampler.py`:
  - Monkeypatch `gm.text.ChatSampler`; pass `state=SimpleNamespace(params=...)`.
- `tests/backend/test_inferencer_path_edge_cases.py`:
  - `_get_checkpoint_path()`: missing dir, empty dir, multiple subdirs -> newest, return `str`.
  - `load_model()` tokenizer/sampler creation failures -> returns False and clears.
- `tests/backend/manager/test_process_manager_private.py`:
  - `_is_process_running()` via `os.kill` stub.
  - `_open_log_files()` OSError path closes partial handle and re-raises.
- `tests/services/test_training_service_wait_timeout.py`:
  - `_wait_for_state_file()` timeout raises; `is_training_running()` fixed “IDLE”.
- `tests/app/components/create_model/test_data_preview.py`:
  - Click “Preview Dataset”: verify raw panel shows DataFrame; tokenized panel renders; treescope/decoder exception surfaced via `st.error`.
- `tests/app/test_main_state_defaults.py`:
  - `_initialize_session_state()` sets all keys; `default_config()` returns sane `TrainingConfig`.

### Robustness edits (minimal changes)
- Inferencer checkpoint path:
  ```diff
  def _get_checkpoint_path(self, work_dir: str) -> str:
-   checkpoint_path = os.path.join(work_dir, CHECKPOINT_SUBDIR)
-   subdirs = [p for p in Path(checkpoint_path).iterdir() if p.is_dir()]
-   return subdirs[0]
+   checkpoint_dir = Path(work_dir) / CHECKPOINT_SUBDIR
+   if not checkpoint_dir.exists():
+       raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
+   subdirs = [p for p in checkpoint_dir.iterdir() if p.is_dir()]
+   if not subdirs:
+       raise FileNotFoundError(f"No checkpoints under: {checkpoint_dir}")
+   subdirs.sort(key=lambda p: p.stat().st_ctime, reverse=True)
+   return str(subdirs[0])
  ```
  - Update `load_model()` to catch and return False; add tests.

- Guard unknown fine-tune method in `ModelTrainer.train()`:
  ```diff
  trainer = FINE_TUNE_STRATEGIES.get(self.training_config.model_config.method)
+ if trainer is None:
+     raise ValueError(f"Unknown fine-tune method: {self.training_config.model_config.method}")
  trainer = trainer.create_trainer(...)
  ```

- Optional: validate `model_variant` against `AppConfig.MODEL_OPTIONS` in `Model.create_standard_model()`.

- Refine `_count_errors()` if needed:
  ```python
  import re
  return len(re.findall(r"\berror\b", stderr_lower))
  ```

- Remove `print(available_charts)` from `system_usage_panel.py`.

- Wrap full preview rendering in try/except in `data_source_selector._show_dataset_preview()` tabs (raw+tokenized).

- Make `TRAINER_MAIN_PATH` absolute using `__file__` resolution.

### Nice-to-have (tooling)
- Run: `pytest --cov=app --cov=backend --cov=services --cov-report=term-missing`
- Add a CI step to fail on uncovered critical modules (loss/sampler/inferencer).

---