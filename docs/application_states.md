# Gemma Fine-Tuning UI - Application States Documentation

## Overview

This document provides a comprehensive overview of all application states, components, and their interactions in the Gemma Fine-tuning UI system. Understanding these states is crucial for maintaining, debugging, and extending the application.

## Application Architecture

### Core Layers

1. **UI Layer** (`app/`)
   - Views: Main page containers
   - Components: Reusable UI elements
   - Services: Business logic abstraction

2. **Backend Layer** (`backend/`)
   - Managers: Process and resource management
   - Core: Training and model functionality
   - Utils: Helper utilities

3. **Configuration Layer** (`config/`)
   - Centralized configuration management
   - Model and training parameters

## Application States

### 1. Global Application States

#### Navigation States (`st.session_state.view`)
- `"welcome"` - Landing/welcome page
- `"create_model"` - Model creation workflow
- `"training_dashboard"` - Training monitoring interface
- `"inference"` - Model testing/inference playground

#### Training Process States
- `session_started_by_app` (boolean) - Tracks if current session initiated training
- `shutdown_requested` (boolean) - Tracks shutdown process initiation
- `abort_confirmation` (boolean) - User confirmation for aborting training

### 2. Training Lifecycle States

#### Process States (File-based)
- **Lock File** (`.training.lock`) - Indicates active training process
- **Status Log** (`status.log`) - Current training status message
- **Process Logs** (`trainer_stdout.log`, `trainer_stderr.log`) - Training output

#### Training Progress States
- **Initialization** - Setting up training environment
- **Active Training** - Model training in progress
- **Completed** - Training finished successfully
- **Failed** - Training terminated with errors
- **Aborted** - Training manually stopped by user

### 3. Component-Level States

#### Control Panel States
The control panel adapts its display based on training status:

**Active Training State:**
```
┌─────────────────────────────────┐
│ Training in progress...         │
│ Status: [Current Status]        │
│ [Abort Training Button]        │
└─────────────────────────────────┘
```

**Failed Training State:**
```
┌─────────────────────────────────┐
│ ❌ Training Failed: [Error]     │
│ [Reset Application Button]     │
│ [Create New Model Button]      │
└─────────────────────────────────┘
```

**Completed Training State:**
```
┌─────────────────────────────────┐
│ ✅ Training concluded           │
│ Latest checkpoint: [Path]      │
│ [Go to Inference] [New Model]  │
└─────────────────────────────────┘
```

**No Checkpoint State:**
```
┌─────────────────────────────────┐
│ ⚠️ Training finished but no     │
│    checkpoint found             │
│ [Reset Application Button]     │
└─────────────────────────────────┘
```

#### KPI Panel States
- **Waiting State** - "Waiting for training data..."
- **Active State** - Displaying real-time metrics:
  - Model Information (parameters, memory, architecture)
  - Training Progress (step, loss, speed, time)
  - Performance Metrics (throughput, ETA, timing)

#### Logs Panel States
- **Waiting State** - "Waiting for training process to start..."
- **Active State** - Real-time log streaming
- **Error State** - Showing error logs with error count metric

### 4. Data Flow States

#### TensorBoard Manager States
- **Cache States**: Fresh, stale, or empty data cache
- **File Monitoring**: Tracking event file modifications
- **Data Processing**: Loading and parsing event data

#### Process Manager States
- **Training Process**: None, running, or terminated
- **TensorBoard Process**: None, running, or terminated
- **Configuration**: Data config and model config storage

## Component Interactions

### 1. State Transitions

#### Training Initiation Flow
```
Welcome Page → Create Model → Training Dashboard (Active)
     ↑                              ↓
     └─── [Abort/Complete] ←────────┘
```

#### Training Lifecycle Flow
```
Initialization → Active → [Completed/Failed/Aborted]
                   ↓              ↓
                Monitor      Terminal State
                   ↑              ↓
                   └──── Reset ←──┘
```

### 2. Service Layer Interactions

#### TrainingService Responsibilities
- Orchestrates process managers
- Abstracts business logic from UI
- Manages training lifecycle
- Provides unified interface for:
  - Starting/stopping training
  - Getting training status
  - Retrieving model configuration
  - Waiting for process initialization

#### Manager Interactions
```
TrainingService
    ├── ProcessManager (process lifecycle)
    ├── TensorBoardManager (metrics/data)
    └── StatusManager (status tracking)
```

### 3. UI State Synchronization

#### Fragment-Based Updates
Several components use `@st.fragment(run_every=1)` for real-time updates:
- `poll_training_status()` - Detects training completion
- `update_tensorboard_event_loop()` - Refreshes metrics data
- `display_control_panel()` - Updates control buttons
- `display_kpi_panel()` - Updates performance metrics
- `display_logs_panel()` - Streams log content

#### State Consistency Mechanisms
- **Training Status Polling** - Detects when training process ends
- **TensorBoard Data Refresh** - Ensures metrics are current
- **Session State Management** - Maintains UI state across reruns

## Error Handling States

### 1. Error Categories

#### Process Errors
- Training process fails to start
- Training process crashes during execution
- TensorBoard fails to initialize
- Resource exhaustion (memory, disk)

#### Configuration Errors
- Invalid model configuration
- Missing or corrupted data
- Incompatible parameter combinations

#### UI Errors
- Session state corruption
- Component rendering failures
- Navigation state conflicts

### 2. Recovery Mechanisms

#### Graceful Degradation
- Show error messages in UI
- Maintain application stability
- Provide recovery options

#### Cleanup Procedures
- Remove stale lock files
- Clear temporary data
- Reset component states

## Monitoring and Debugging

### 1. State Inspection Points

#### File-Based State
- Check lock file existence: `os.path.exists(config.LOCK_FILE)`
- Read status log: `status_manager.get()`
- Monitor process logs for errors

#### Session State
- `st.session_state.session_started_by_app`
- `st.session_state.view`
- Component-specific states

#### Process State
- Training process PID and status
- TensorBoard server status
- Resource utilization

### 2. Common Debug Scenarios

#### Orphaned Sessions
**Symptoms**: Lock file exists but no UI session owns it
**Resolution**: Show orphan recovery UI, allow monitoring or cleanup

#### State Desynchronization
**Symptoms**: UI shows incorrect state vs actual process state
**Resolution**: Force state refresh, validate against file system

#### Resource Conflicts
**Symptoms**: Multiple training attempts, port conflicts
**Resolution**: Proper cleanup, resource checking

## Configuration States

### 1. Static Configuration (AppConfig)
- Model information and options
- Training parameters and defaults
- File paths and system settings
- Fine-tuning method definitions

### 2. Runtime Configuration
- User-selected model parameters
- Data source configuration
- Training-specific settings
- Component preferences

### 3. Configuration Persistence
- Model config stored in ProcessManager during training
- Available via `training_service.get_model_config()`
- Cleared when training process ends

## Performance Considerations

### 1. State Update Frequency
- Fragment updates: Every 1 second
- TensorBoard refresh: On-demand with caching
- Log streaming: Real-time file monitoring

### 2. Resource Management
- Automatic cleanup on process termination
- Memory-efficient data caching
- Proper file handle management

### 3. Scalability Factors
- Single training process per application instance
- TensorBoard data size growth over time
- Session state memory usage

## Future State Management Improvements

### 1. State Persistence
- Database-backed state storage
- Training history preservation
- User preference persistence

### 2. Multi-Session Support
- Concurrent training support
- Session isolation
- Resource allocation

### 3. Enhanced Monitoring
- Health check endpoints
- Structured logging
- Metrics export

## Conclusion

This comprehensive state documentation provides the foundation for understanding how the Gemma Fine-tuning UI manages complex interactions between UI components, training processes, and data flow. Understanding these states is essential for effective maintenance, debugging, and feature development. 