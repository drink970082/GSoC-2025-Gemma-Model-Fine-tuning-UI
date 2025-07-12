# Training Lifecycle Analysis: Status, Session State, and File Management

## Overview

This document provides a comprehensive analysis of the training lifecycle in the Gemma Fine-Tuning UI, covering all possible scenarios, state transitions, file management, and session state variables. This analysis is crucial for understanding, debugging, and maintaining the application. (synthesis)

## 1. Training Status Enum Values

```python
class TrainingStatus(Enum):  # config/app_config.py:6
    IDLE = auto()        # No training active, ready to start
    RUNNING = auto()     # Training process actively running
    FINISHED = auto()    # Training completed successfully
    FAILED = auto()      # Training failed with errors
    ORPHANED = auto()    # Lock file exists but process is dead
```
(config/app_config.py:6-12)

## 2. Session State Variables

### Core Session State (app/main.py)
```python
st.session_state.view                    # app/main.py:23, used throughout views
st.session_state.abort_training          # app/main.py:25, used in dashboard, control_panel, etc.
st.session_state.session_started_by_app  # app/main.py:27, used in dashboard, control_panel, etc.
st.session_state.sampler                 # app/main.py:29, set in checkpoint_selection.py:128
st.session_state.tokenizer               # app/main.py:31, set in checkpoint_selection.py:129
```

### Frozen State (for aborted training)
```python
st.session_state.frozen_kpi_data         # app/main.py:33, used in kpi_panel.py:72,75
st.session_state.frozen_log              # app/main.py:35, used in logs_panel.py:12,27
st.session_state.frozen_loss_metrics     # app/main.py:37, used in plots_panel.py:64,69
st.session_state.frozen_perf_metrics     # app/main.py:39, used in plots_panel.py:65,70
```

### Additional Session State (used in specific views)
```python
st.session_state.abort_confirmation      # app/view/welcome_view.py:28,48,55,59
```

## 3. File Management System

### Lock File (`.training.lock`)
- **Purpose**: Single source of truth for training state (backend/manager/process_manager.py:21, 57, 252)
- **Content**: Process ID of training process (backend/manager/process_manager.py:117, 137)
- **Location**: Work directory (created directly in work directory) (backend/manager/process_manager.py:297)
- **Lifecycle**: Created when training starts, removed when training ends (backend/manager/process_manager.py:57, 117, 137, 284)

### Work Directory Structure
```
checkpoints/
└── {model_name}-{timestamp}/
    ├── .training.lock          # Lock file (see above)
    ├── status.log              # backend/manager/status_manager.py:11, 19
    ├── trainer_stdout.log      # backend/manager/process_manager.py:297
    ├── trainer_stderr.log      # backend/manager/process_manager.py:297
    ├── model_config.json       # backend/manager/process_manager.py:52, 117
    ├── checkpoints/            # output dir for model checkpoints
    └── events.out.tfevents     # TensorBoard events (noted in docs)
```

### Critical File Operations
```python
def _write_lock_file(self, pid: int) -> None:      # backend/manager/process_manager.py:117
def _read_lock_file(self) -> int | None:           # backend/manager/process_manager.py:137
def _remove_lock_file(self) -> None:               # backend/manager/process_manager.py:147
def is_lock_file_locked(self) -> bool:             # backend/manager/process_manager.py:156
```

## 4. Scenario Analysis

### Scenario 1: Fresh Start (No Previous Training)

**Initial State:**
- `TrainingStatus.IDLE` (config/app_config.py:7)
- No lock file exists (backend/manager/process_manager.py:156)
- All session state variables initialized to defaults (app/main.py:23-39)

**Process:**
1. User clicks "Start Fine-tuning" (app/view/create_model_view.py:68)
2. `training_service.start_training()` called (app/view/create_model_view.py:69)
3. Work directory created: `{model_name}-{timestamp}/` (services/training_service.py:144)
4. Training subprocess started (backend/manager/process_manager.py:57)
5. Lock file created with PID in work directory (backend/manager/process_manager.py:117)
6. `wait_for_lock_file()` waits for lock file (services/training_service.py:49)
7. Redirect to training dashboard (app/view/create_model_view.py:70)

**Final State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Lock file exists with PID (backend/manager/process_manager.py:117)
- `session_started_by_app = True` (app/view/create_model_view.py:35)

**Files Created:**
- `.training.lock` (in work directory) (backend/manager/process_manager.py:117)
- `status.log` (backend/manager/status_manager.py:11, 19)
- `trainer_stdout.log` (backend/manager/process_manager.py:297)
- `trainer_stderr.log` (backend/manager/process_manager.py:297)
- `model_config.json` (backend/manager/process_manager.py:52, 117)

### Scenario 2: Training Completion (Success)

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Lock file exists (backend/manager/process_manager.py:156)
- Training process running (backend/manager/process_manager.py:252)

**Process:**
1. Training process completes successfully (exit code 0) (backend/manager/process_manager.py:255)
2. `poll_training_status()` fragment detects completion (app/view/training_dashboard_view.py:15-35)
3. `process_manager.get_status()` returns `TrainingStatus.FINISHED` (backend/manager/process_manager.py:259)
4. `training_service.is_training_running()` calls `reset_state()` (services/training_service.py:81)
5. Lock file removed (backend/manager/process_manager.py:147, 284)
6. Work directory preserved (contains checkpoints) (backend/manager/process_manager.py:284)

**Final State:**
- `TrainingStatus.IDLE` (backend/manager/process_manager.py:272)
- No lock file (backend/manager/process_manager.py:156)
- Checkpoints available in work directory (backend/inferencer.py:25-30)
- `session_started_by_app = False` (app/view/training_dashboard_view.py:28)

**Files Status:**
- `.training.lock` - Removed (backend/manager/process_manager.py:147, 284)
- Work directory - Preserved with checkpoints (backend/manager/process_manager.py:284)
- Log files - Available for review (backend/manager/process_manager.py:240-250)

### Scenario 3: Training Failure

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Lock file exists (backend/manager/process_manager.py:156)
- Training process running (backend/manager/process_manager.py:252)

**Process:**
1. Training process fails (non-zero exit code) (backend/manager/process_manager.py:261)
2. `poll_training_status()` detects failure (app/view/training_dashboard_view.py:15-35)
3. `process_manager.get_status()` returns `TrainingStatus.FAILED` (backend/manager/process_manager.py:263)
4. `training_service.is_training_running()` calls `reset_state(delete_checkpoint=True)` (services/training_service.py:84)
5. Lock file removed (backend/manager/process_manager.py:147, 284)
6. Work directory deleted (no checkpoints saved) (backend/manager/process_manager.py:284)

**Final State:**
- `TrainingStatus.IDLE` (backend/manager/process_manager.py:272)
- No lock file (backend/manager/process_manager.py:156)
- No work directory (deleted) (backend/manager/process_manager.py:284)
- `session_started_by_app = False` (app/view/training_dashboard_view.py:28)

**Files Status:**
- `.training.lock` - Removed (backend/manager/process_manager.py:147, 284)
- Work directory - Deleted (backend/manager/process_manager.py:284)
- No checkpoints saved (backend/manager/process_manager.py:284)

### Scenario 4: User Abort Training

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Lock file exists (backend/manager/process_manager.py:156)
- Training process running (backend/manager/process_manager.py:252)

**Process:**
1. User clicks "Abort Training" (app/components/training_dashboard/control_panel.py:23)
2. `training_service.stop_training(mode="graceful")` called (app/components/training_dashboard/control_panel.py:24)
3. SIGINT sent to training process (backend/manager/process_manager.py:125)
4. If graceful fails, `stop_training(mode="force")` called (backend/manager/process_manager.py:130)
5. SIGKILL sent to training process (backend/manager/process_manager.py:133)
6. `reset_state(delete_checkpoint=True)` called (services/training_service.py:84)
7. Lock file removed (backend/manager/process_manager.py:147, 284)
8. Work directory deleted (backend/manager/process_manager.py:284)

**Final State:**
- `TrainingStatus.IDLE` (backend/manager/process_manager.py:272)
- No lock file (backend/manager/process_manager.py:156)
- No work directory (backend/manager/process_manager.py:284)
- `abort_training = True` (app/components/training_dashboard/control_panel.py:24)
- `session_started_by_app = False` (app/view/training_dashboard_view.py:28)

**Files Status:**
- `.training.lock` - Removed (backend/manager/process_manager.py:147, 284)
- Work directory - Deleted (backend/manager/process_manager.py:284)
- Frozen state variables set for UI display (app/components/training_dashboard/kpi_panel.py:75, logs_panel.py:27, plots_panel.py:69,70)

### Scenario 5: Orphaned Process Detection

**Initial State:**
- Lock file exists from previous session (backend/manager/process_manager.py:156)
- Process is dead (backend/manager/process_manager.py:162)
- New session starts (synthesis)

**Process:**
1. `training_service.is_training_running()` called (services/training_service.py:81)
2. `process_manager.get_status()` detects orphaned state (backend/manager/process_manager.py:252, 262)
3. `TrainingStatus.ORPHANED` returned (backend/manager/process_manager.py:262)
4. Warning message displayed: "An orphaned training process from a previous session was detected. Automatically cleaning up...." (services/training_service.py:87)
5. `process_manager.force_cleanup()` called (services/training_service.py:88, backend/manager/process_manager.py:177)
6. Lock file removed (backend/manager/process_manager.py:147, 284)
7. Any remaining processes killed via `pkill` (backend/manager/process_manager.py:178)

**Final State:**
- `TrainingStatus.IDLE` (backend/manager/process_manager.py:272)
- No lock file (backend/manager/process_manager.py:156)
- Clean state ready for new training (backend/manager/process_manager.py:284-295)

**Files Status:**
- `.training.lock` - Removed (backend/manager/process_manager.py:147, 284)
- Orphaned processes - Killed (backend/manager/process_manager.py:178)
- Clean state achieved (backend/manager/process_manager.py:284-295)

### Scenario 6: Second Training Run (Common Issue)

**Initial State:**
- Previous training completed/failed/aborted (backend/manager/process_manager.py:259, 263)
- Stale lock file may exist (backend/manager/process_manager.py:156)
- Session state variables set (app/main.py:23-39)

**Process:**
1. User starts new training (app/view/create_model_view.py:68)
2. `training_service.start_training()` called (app/view/create_model_view.py:69)
3. **ISSUE**: Stale lock file not properly cleaned up (backend/manager/process_manager.py:57-67)
4. `wait_for_lock_file()` waits indefinitely (services/training_service.py:49)
5. UI hangs on "Waiting for training process to initialize..." (app/view/create_model_view.py:65)

**Root Cause:**
- Lock file cleanup logic in `is_training_running()` may not handle all edge cases (services/training_service.py:81)
- Race condition between status checks and cleanup (backend/manager/process_manager.py:57-67)
- Stale lock files from previous sessions not properly cleaned up (backend/manager/process_manager.py:147)

**Current Implementation:**
```python
def start_training(self, training_config: TrainingConfig) -> None:
    if self.is_training_running() == TrainingStatus.RUNNING:
        st.warning("Training is already in progress.")
        return
    # No explicit stale lock file cleanup implemented
```
(services/training_service.py:144-148)

**Final State:**
- `TrainingStatus.RUNNING` (if successful) (backend/manager/process_manager.py:252)
- Lock file with PID in work directory (backend/manager/process_manager.py:117)
- New training session started (app/view/create_model_view.py:70)

## 5. User Interaction Scenarios

### Scenario 7: Browser Refresh During Training

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- Session state variables set (app/main.py:23-39)

**Process:**
1. User refreshes browser (F5 or Ctrl+R) (synthesis)
2. Streamlit session state is reset to defaults (synthesis)
3. `_register_session_state()` reinitializes session variables (app/main.py:41-47)
4. `training_service.is_training_running()` called (app/main.py:47)
5. Process manager detects running training via lock file (backend/manager/process_manager.py:156)
6. Status determined from file system state (backend/manager/process_manager.py:252)

**Final State:**
- `TrainingStatus.RUNNING` (detected from lock file) (backend/manager/process_manager.py:252)
- Session state reinitialized (app/main.py:41-47)
- Training continues uninterrupted (backend/manager/process_manager.py:252)
- User can monitor training in dashboard (app/view/training_dashboard_view.py:15-35)

**Recovery:**
- Automatic detection of running training (backend/manager/process_manager.py:252)
- Session state reconstruction (app/main.py:41-47)
- Seamless continuation of monitoring (app/view/training_dashboard_view.py:15-35)

### Scenario 8: Browser Tab Close and Reopen

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- Session state in browser memory (synthesis)

**Process:**
1. User closes browser tab/window (synthesis)
2. Session state lost in browser (synthesis)
3. Training process continues running (backend/manager/process_manager.py:252)
4. User reopens application in new tab (synthesis)
5. New session created with default state (app/main.py:41-47)
6. `training_service.is_training_running()` detects running training (app/main.py:47)

**Final State:**
- `TrainingStatus.RUNNING` (detected from lock file) (backend/manager/process_manager.py:252)
- New session with reinitialized state (app/main.py:41-47)
- Training continues uninterrupted (backend/manager/process_manager.py:252)
- User can resume monitoring (app/view/training_dashboard_view.py:15-35)

**Recovery:**
- File-based state detection (backend/manager/process_manager.py:156)
- Session state reconstruction (app/main.py:41-47)
- Training process unaffected (backend/manager/process_manager.py:252)

### Scenario 9: Multiple Browser Tabs

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- Multiple browser tabs open (synthesis)

**Process:**
1. User has multiple tabs open to the application (synthesis)
2. Each tab has independent session state (synthesis)
3. Training status checked independently in each tab (app/main.py:47)
4. All tabs show same training status (from file system) (backend/manager/process_manager.py:252)
5. Actions in one tab affect all tabs (synthesis)

**Final State:**
- All tabs show consistent training status (backend/manager/process_manager.py:252)
- Training process continues normally (backend/manager/process_manager.py:252)
- Potential for conflicting user actions (app/components/training_dashboard/control_panel.py:23-24)

**Issues:**
- Race conditions if multiple users interact simultaneously (app/components/training_dashboard/control_panel.py:23-24)
- Session state inconsistency across tabs (app/main.py:23-39)
- Need for proper synchronization (app/main.py:47)

### Scenario 10: Network Disconnection

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- User monitoring dashboard (synthesis)

**Process:**
1. Network connection lost (synthesis)
2. Streamlit fragments stop updating (synthesis)
3. Training process continues running (backend/manager/process_manager.py:252)
4. Network reconnects (synthesis)
5. Fragments resume updating (synthesis)
6. Status synchronized from file system (backend/manager/process_manager.py:252)

**Final State:**
- `TrainingStatus.RUNNING` (or current actual status) (backend/manager/process_manager.py:252)
- Monitoring resumes automatically (app/view/training_dashboard_view.py:15-35)
- Training process unaffected (backend/manager/process_manager.py:252)

**Recovery:**
- Automatic reconnection handling (app/view/training_dashboard_view.py:15-35)
- Status synchronization on reconnect (backend/manager/process_manager.py:252)
- No training interruption (backend/manager/process_manager.py:252)

## 6. Application Lifecycle Scenarios

### Scenario 11: Streamlit App Graceful Shutdown (Ctrl+C)

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- Streamlit app running (synthesis)

**Process:**
1. User presses Ctrl+C in terminal (synthesis)
2. Streamlit receives SIGINT signal (synthesis)
3. `atexit` handlers trigger cleanup via `di_container._cleanup_all()` (synthesis)
4. `process_manager.cleanup()` called (graceful termination only) (synthesis)
5. Training process may continue running if graceful shutdown fails (synthesis)
6. Lock file may remain if process doesn't terminate (synthesis)

**Final State:**
- Training process may continue running (synthesis)
- Lock file may remain on disk (synthesis)
- Application exits (synthesis)
- Orphaned process detection needed on restart (services/training_service.py:87)

**Recovery:**
- Orphaned process detection on next startup (services/training_service.py:87)
- Manual cleanup may be required (synthesis)
- User can restart application (synthesis)

### Scenario 12: Streamlit App Crash/Unexpected Exit

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- Streamlit app running (synthesis)

**Process:**
1. Streamlit app crashes (memory error, exception, etc.) (synthesis)
2. No cleanup handlers executed (synthesis)
3. Training process continues running (backend/manager/process_manager.py:252)
4. Lock file remains on disk (synthesis)
5. User restarts application (synthesis)

**Final State:**
- Training process continues running (backend/manager/process_manager.py:252)
- Lock file exists (synthesis)
- New Streamlit session starts (app/main.py:41-47)

**Recovery:**
- Orphaned process detection on startup (services/training_service.py:87)
- User can choose to monitor or abort training (synthesis)
- Manual cleanup may be required (synthesis)

### Scenario 13: System Reboot/Server Restart

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- Application running (synthesis)

**Process:**
1. System reboots or server restarts (synthesis)
2. All processes terminated (synthesis)
3. Training process killed (synthesis)
4. Lock file remains on disk (synthesis)
5. Application restarts after reboot (synthesis)

**Final State:**
- Training process dead (synthesis)
- Lock file exists but stale (synthesis)
- Application detects orphaned state (services/training_service.py:87)

**Recovery:**
- Orphaned process detection (services/training_service.py:87)
- Automatic cleanup of stale lock file (backend/manager/process_manager.py:177)
- User can restart training (synthesis)

### Scenario 14: Docker Container Restart

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- Application in Docker container (synthesis)

**Process:**
1. Docker container restarts (synthesis)
2. All processes in container terminated (synthesis)
3. Training process killed (synthesis)
4. Lock file persists if volume mounted (synthesis)
5. Container restarts with same volume (synthesis)

**Final State:**
- Training process dead (synthesis)
- Lock file exists (if volume mounted) (synthesis)
- Application detects orphaned state (services/training_service.py:87)

**Recovery:**
- Orphaned process detection (services/training_service.py:87)
- Automatic cleanup (backend/manager/process_manager.py:177)
- Volume persistence considerations (synthesis)

### Scenario 15: Resource Exhaustion (Memory/Disk)

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- System resources normal (synthesis)

**Process:**
1. Memory or disk space exhausted (synthesis)
2. Training process may crash or hang (synthesis)
3. Streamlit app may become unresponsive (synthesis)
4. System may kill processes (synthesis)
5. Lock file may become stale (synthesis)

**Final State:**
- Training process dead or hanging (synthesis)
- Lock file may exist (synthesis)
- Application state uncertain (synthesis)

**Recovery:**
- Resource monitoring and alerts (synthesis)
- Automatic cleanup on resource recovery (synthesis)
- User intervention required (synthesis)

## 7. Advanced User Interaction Scenarios

### Scenario 18: Browser Back/Forward Navigation

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- User on training dashboard (synthesis)

**Process:**
1. User clicks browser back button (synthesis)
2. Navigates to previous page (create model, welcome) (synthesis)
3. Training continues running (backend/manager/process_manager.py:252)
4. User clicks forward button (synthesis)
5. Returns to training dashboard (synthesis)
6. Status rechecked from file system (backend/manager/process_manager.py:252)

**Final State:**
- `TrainingStatus.RUNNING` (detected from lock file) (backend/manager/process_manager.py:252)
- Training continues uninterrupted (backend/manager/process_manager.py:252)
- Dashboard shows current status (app/view/training_dashboard_view.py:15-35)

**Recovery:**
- Automatic status detection on page load (app/main.py:47)
- Seamless navigation experience (app/view/training_dashboard_view.py:15-35)
- No training interruption (backend/manager/process_manager.py:252)

### Scenario 19: Browser Bookmark Access

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- User has bookmarked training dashboard (synthesis)

**Process:**
1. User accesses bookmarked URL directly (synthesis)
2. New session created (app/main.py:41-47)
3. `training_service.is_training_running()` called (app/main.py:47)
4. Status determined from file system (backend/manager/process_manager.py:252)
5. Appropriate page displayed based on status (synthesis)

**Final State:**
- Correct page displayed based on training status (app/main.py:47)
- Session state initialized properly (app/main.py:41-47)
- Training continues uninterrupted (backend/manager/process_manager.py:252)

**Recovery:**
- Direct URL access handling (app/main.py:47)
- Status-based page routing (app/main.py:47)
- Session state initialization (app/main.py:41-47)


### Scenario 21: Incognito/Private Browsing

**Initial State:**
- `TrainingStatus.RUNNING` (backend/manager/process_manager.py:252)
- Training process active (backend/manager/process_manager.py:252)
- User opens incognito window (synthesis)

**Process:**
1. User opens incognito/private browsing window (synthesis)
2. New isolated session created (synthesis)
3. No session state persistence (synthesis)
4. Training status checked from file system (backend/manager/process_manager.py:252)
5. Fresh session state initialized (app/main.py:41-47)

**Final State:**
- Isolated session with fresh state (app/main.py:41-47)
- Training continues running (backend/manager/process_manager.py:252)
- No session state sharing with main browser (app/main.py:23-39)

**Recovery:**
- File-based state detection (backend/manager/process_manager.py:156)
- Session isolation maintained (app/main.py:23-39)
- Training process unaffected (backend/manager/process_manager.py:252)


## 9. Summary and Best Practices

### Key Insights
- File-based state management provides resilience (backend/manager/process_manager.py:156)
- Session state is ephemeral and browser-dependent (app/main.py:23-39)
- Training process independence ensures continuity (backend/manager/process_manager.py:252)
- Orphaned process detection handles edge cases (services/training_service.py:87)

### Recommended Practices
- Always check file system state for training status (backend/manager/process_manager.py:252)
- Implement proper cleanup on application shutdown (synthesis)
- Handle stale lock files gracefully (backend/manager/process_manager.py:177)
- Provide clear user feedback for all scenarios (synthesis)
- Monitor and log all state transitions (synthesis)

### Common Pitfalls
- Relying solely on session state for training status (synthesis)
- Not handling orphaned processes properly (synthesis)
- Missing cleanup in error scenarios (synthesis)
- Inconsistent state across multiple browser tabs (synthesis)
- Poor error handling for network issues (synthesis)

### Future Improvements
- Implement proper user session management (synthesis)
- Add comprehensive logging and monitoring (synthesis)
- Improve error recovery mechanisms (synthesis)
- Enhance concurrent access handling (synthesis)
- Add automated testing for edge cases (synthesis)