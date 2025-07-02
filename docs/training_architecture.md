# Gemma Fine-Tuning Architecture

## System Overview

The Gemma Fine-Tuning system provides a streamlined interface for monitoring and controlling the fine-tuning process of Gemma language models. The system uses a modern service-oriented architecture with clear separation of concerns between UI, business logic, and backend operations.

## Current Architecture (Post-Refactoring)

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    UI Layer (Streamlit)                    │
├─────────────────────────────────────────────────────────────┤
│ • Views (main.py, create_model_view.py, etc.)             │
│ • Components (training_dashboard/, create_model/, etc.)    │
│ • Fragment-based real-time updates                        │
└─────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────┐
│                   Service Layer                            │
├─────────────────────────────────────────────────────────────┤
│ • TrainingService (business logic orchestration)          │
│ • Global service management (global_service.py)           │
│ • Abstraction from backend complexity                     │
└─────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────┐
│                  Backend Layer                             │
├─────────────────────────────────────────────────────────────┤
│ • Managers (Process, TensorBoard, Status)                 │
│ • Core Training Engine (trainer.py, model.py)             │
│ • Global manager coordination (global_manager.py)         │
└─────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────┐
│                Configuration Layer                         │
├─────────────────────────────────────────────────────────────┤
│ • AppConfig (centralized configuration)                   │
│ • Model and data configurations                           │
│ • Runtime parameter management                            │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Service Layer (NEW)

#### TrainingService
**Location**: `app/services/training_service.py`
**Purpose**: Central orchestrator for all training-related operations

```python
class TrainingService:
    def start_training(self, app_config: dict) -> None
    def stop_training(self, mode: Literal["graceful", "force"]) -> bool
    def is_training_running(self) -> bool
    def get_training_status(self) -> str
    def get_model_config(self) -> dict | None
```

**Key Responsibilities**:
- Abstracts complex manager interactions
- Provides unified interface for UI components
- Manages training lifecycle
- Handles configuration persistence

### 2. Manager Layer (Backend)

#### ProcessManager
**Location**: `backend/manager/process_manager.py`
**Purpose**: Manages subprocess lifecycle and resource cleanup

**Key Features**:
- Training process management (start/stop/monitor)
- TensorBoard server management
- Graceful and forced shutdown capabilities
- Configuration storage during training
- Automatic cleanup on application exit

#### TensorBoardDataManager
**Location**: `backend/manager/tensorboard_manager.py`
**Purpose**: Handles TensorBoard data parsing and caching

**Key Features**:
- Event file monitoring and parsing
- Data caching with time-based invalidation
- Real-time metrics extraction
- Performance optimization for large datasets

#### StatusManager
**Location**: `backend/manager/status_manager.py`
**Purpose**: Manages training status communication

**Key Features**:
- File-based status persistence
- Status message updates from training process
- Cleanup and recovery operations

### 3. UI Component Architecture

#### Fragment-Based Updates
Real-time UI updates using Streamlit fragments:

```python
@st.fragment(run_every=1)
def poll_training_status():
    # Detects training completion and updates state

@st.fragment(run_every=1)
def update_tensorboard_event_loop():
    # Refreshes TensorBoard data cache

@st.fragment(run_every=1)
def display_control_panel():
    # Updates control buttons based on current state
```

#### Component Organization
- **Control Panel**: Training state management and user actions
- **KPI Panel**: Real-time performance metrics and model information
- **Logs Panel**: Live log streaming with error detection
- **Plots Panel**: Interactive TensorBoard visualizations
- **System Usage Panel**: Resource monitoring and utilization

## Training Process Flow

### 1. Training Initiation
```
User Input → CreateModelView → TrainingService.start_training()
    ↓
ProcessManager.update_config() → ProcessManager.start_training()
    ↓
Training Process Launch → Lock File Creation → Status Updates
```

### 2. Active Training Monitoring
```
Fragment Updates (1s intervals)
    ↓
TrainingService.get_training_status()
    ↓
TensorBoardManager.get_data() → UI Component Updates
    ↓
Real-time Dashboard Display
```

### 3. Training Completion/Termination
```
Training Process Ends → Lock File Removal
    ↓
Fragment Detection → State Update
    ↓
Control Panel Update → User Notification
    ↓
Cleanup Operations
```

## State Management

### Session State Variables
- `st.session_state.view`: Current application view
- `st.session_state.session_started_by_app`: Session ownership tracking
- `st.session_state.shutdown_requested`: Shutdown process state

### File-Based State
- `.training.lock`: Active training indicator
- `status.log`: Current training status
- `trainer_stdout.log` / `trainer_stderr.log`: Process output

### Configuration Persistence
- Model configuration stored in ProcessManager during training
- Retrieved via `TrainingService.get_model_config()`
- Cleared when training process ends

## Data Flow Patterns

### 1. Configuration Flow
```
UI Input → DataConfig/ModelConfig → TrainingService
    ↓
ProcessManager.update_config() → Training Process
```

### 2. Status Flow
```
Training Process → StatusManager.update() → File System
    ↓
TrainingService.get_training_status() → UI Display
```

### 3. Metrics Flow
```
Training Process → TensorBoard Events → File System
    ↓
TensorBoardManager.get_data() → Cache → UI Components
```

## Error Handling and Recovery

### Error Categories
1. **Process Errors**: Training/TensorBoard process failures
2. **Configuration Errors**: Invalid parameters or missing data
3. **Resource Errors**: Insufficient memory or disk space
4. **Network Errors**: TensorBoard connectivity issues

### Recovery Mechanisms
1. **Graceful Shutdown**: SIGINT followed by cleanup
2. **Forced Shutdown**: SIGKILL with orphan process cleanup
3. **Session Recovery**: Detect orphaned training processes
4. **State Reset**: Clean application state and file system

## Performance Considerations

### Fragment Optimization
- **Update Frequency**: 1-second intervals for real-time feel
- **Conditional Updates**: Only update when data changes
- **Data Caching**: TensorBoard data cached with TTL

### Resource Management
- **Memory Efficiency**: Periodic cache clearing
- **Process Cleanup**: Automatic cleanup on exit
- **File Monitoring**: Efficient file change detection

### Scalability Factors
- **Single Training Process**: One active training per application instance
- **TensorBoard Data Growth**: Cached data size monitoring
- **Session State Size**: Minimal session state usage

## Current Limitations and Technical Debt

### Dependency Management
- **Singleton Patterns**: Multiple global singletons create coupling
- **Global State**: Heavy reliance on global managers
- **Import Dependencies**: Circular dependency potential

### Architecture Issues
- **Mixed Responsibilities**: Some components handle both UI and logic
- **Session State Overuse**: Application state mixed with UI state
- **Error Handling**: Inconsistent error handling patterns

### Performance Issues
- **Fragment Overhead**: Multiple fragments create UI lag
- **Memory Leaks**: Potential memory leaks with singleton patterns
- **File I/O**: Inefficient polling and reading patterns

## Future Architecture Improvements

### Planned Refactoring
1. **Dependency Injection**: Replace singletons with DI container
2. **State Management**: Centralized, type-safe state management
3. **Error Handling**: Comprehensive error handling framework
4. **Testing**: Full test coverage with mocked dependencies

### Architecture Modernization
1. **Event-Driven Design**: Loose coupling through event bus
2. **Plugin Architecture**: Extensible component system
3. **Configuration Management**: Environment-based configuration
4. **Monitoring**: Health checks and metrics export

## Best Practices

### Development Guidelines
1. **Service Layer**: All business logic through TrainingService
2. **Error Handling**: Consistent error patterns and user feedback
3. **State Management**: Minimal session state, prefer service state
4. **Component Design**: Single responsibility, clear interfaces

### Monitoring and Debugging
1. **State Inspection**: File-based state for debugging
2. **Process Monitoring**: Health checks and resource tracking
3. **Error Logging**: Comprehensive error logging and reporting
4. **Performance Metrics**: Response time and resource usage monitoring

## Integration Points

### External Dependencies
- **Kauldron**: Training framework integration
- **TensorBoard**: Metrics visualization and logging
- **Gemma Models**: Model loading and checkpoint management
- **Data Pipeline**: HuggingFace, TensorFlow, custom data sources

### System Dependencies
- **File System**: Configuration, logs, and checkpoint storage
- **Process Management**: Subprocess creation and management
- **Network**: TensorBoard server and port management
- **Memory**: Model loading and training data management

This updated architecture documentation reflects the current state of the system after implementing the service layer refactoring, while also highlighting areas for future improvement identified in the refactoring roadmap. 