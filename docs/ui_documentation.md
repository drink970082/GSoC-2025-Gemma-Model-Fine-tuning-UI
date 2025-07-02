# Gemma Fine-tuning UI Documentation

## Overview

The Gemma Fine-tuning UI is a comprehensive web interface for fine-tuning Google's Gemma language models. Built with Streamlit, it provides an intuitive workflow for creating, training, monitoring, and testing fine-tuned models with real-time updates and professional-grade monitoring capabilities.

## Application Architecture

### Service-Oriented Design
The UI follows a modern service-oriented architecture with clear separation of concerns:

- **UI Layer**: Streamlit views and components
- **Service Layer**: Business logic abstraction (`TrainingService`)
- **Backend Layer**: Process and data management
- **Configuration Layer**: Centralized configuration management

### Real-Time Updates
The interface uses Streamlit's fragment system for real-time updates without full page refreshes:

```python
@st.fragment(run_every=1)
def update_component():
    # Real-time component updates
```

## Main Features

### 1. Welcome/Navigation System

**Location**: `app/main.py`
**Purpose**: Application entry point and navigation control

**Features**:
- **Dynamic Navigation**: Intelligent routing based on training state
- **Session Recovery**: Handles interrupted sessions and orphaned processes
- **State Persistence**: Maintains application state across browser refreshes

**Navigation States**:
- `welcome`: Landing page for new sessions
- `create_model`: Model configuration workflow
- `training_dashboard`: Active training monitoring
- `inference`: Model testing playground

### 2. Model Creation Workflow

**Location**: `app/view/create_model_view.py`
**Components**: `app/components/create_model/`

#### Configuration Steps
1. **Model Selection**
   - Gemma 2 models: 2B, 9B, 27B parameters
   - Gemma 3 models: 1B, 4B, 12B, 27B parameters
   - Fine-tuning method selection (Standard, LoRA, DPO)

2. **Data Source Configuration**
   - **HuggingFace Datasets**: Direct integration with HF Hub
   - **TensorFlow Datasets**: TFDS compatibility
   - **Custom JSON**: Upload and validate custom training data
   - **Data Preview**: Real-time data validation and preview

3. **Training Parameters**
   - Epochs, batch size, learning rate configuration
   - Method-specific parameters (LoRA rank/alpha, DPO beta)
   - Resource allocation settings

4. **Configuration Review**
   - Complete parameter summary before training
   - Validation and error checking
   - One-click training initiation

### 3. Training Dashboard

**Location**: `app/view/training_dashboard_view.py`
**Components**: `app/components/training_dashboard/`

#### Real-Time Components

##### Control Panel
**Location**: `app/components/training_dashboard/control_panel.py`

**Features**:
- **Dynamic State Display**: Adapts to current training state
- **Action Buttons**: Context-sensitive controls (Abort, Reset, Create New)
- **Status Messages**: Clear communication of current state
- **Graceful Shutdown**: Two-stage shutdown process (graceful → forced)

**States**:
- **Active Training**: Status display + Abort button
- **Failed Training**: Error message + Reset/Create New buttons
- **Completed Training**: Success message + Inference/Create New buttons
- **No Checkpoint**: Warning message + Reset button

##### KPI Panel
**Location**: `app/components/training_dashboard/kpi_panel.py`

**Metrics Displayed**:
- **Model Information**: Parameters, memory usage, architecture details
- **Training Progress**: Current step, loss value, completion percentage
- **Performance Metrics**: Training speed, data throughput, ETA
- **Time Tracking**: Total training time, average step time

**Features**:
- **Real-time Updates**: 1-second refresh intervals
- **Smart Formatting**: Human-readable numbers and time formats
- **Error Handling**: Graceful degradation when data unavailable

##### Logs Panel
**Location**: `app/components/training_dashboard/logs_panel.py`

**Features**:
- **Live Log Streaming**: Real-time stdout/stderr display
- **Error Detection**: Automatic error highlighting and counting
- **Expandable Interface**: Collapsible log sections
- **Auto-scroll**: Automatic scrolling to latest content

##### Plots Panel
**Location**: `app/components/training_dashboard/plots_panel.py`

**Visualizations**:
- **Interactive Charts**: Plotly-based interactive visualizations
- **TensorBoard Integration**: Embedded TensorBoard iframe
- **Loss Curves**: Real-time loss progression
- **Performance Metrics**: Training speed and resource usage over time

##### System Usage Panel
**Location**: `app/components/training_dashboard/system_usage_panel.py`

**Monitoring**:
- **Resource Utilization**: CPU, memory, GPU usage
- **Real-time Metrics**: Live system monitoring
- **Historical Tracking**: Resource usage over time
- **Alert System**: Warnings for resource constraints

### 4. Inference Playground

**Location**: `app/view/inference_view.py`
**Components**: `app/components/inference/`

**Features**:
- **Model Testing**: Test fine-tuned models with custom prompts
- **Response Analysis**: Evaluate model outputs and behavior
- **Performance Comparison**: Compare base vs fine-tuned model responses
- **Interactive Interface**: Real-time model interaction

## Technical Implementation

### Component Architecture

#### Fragment-Based Updates
Modern real-time UI with minimal overhead:

```python
# Training status monitoring
@st.fragment(run_every=1)
def poll_training_status():
    training_service = get_training_service()
    if not training_service.is_training_running():
        if st.session_state.session_started_by_app:
            st.session_state.session_started_by_app = False
            st.rerun()

# Data refresh
@st.fragment(run_every=1)
def update_tensorboard_event_loop():
    get_tensorboard_manager().get_data()
```

#### Service Integration
All UI components interact through the service layer:

```python
from app.services.global_service import get_training_service

training_service = get_training_service()
# Standardized service interactions
training_service.start_training(config)
training_service.is_training_running()
training_service.get_training_status()
```

### State Management

#### Session State Usage
Minimal session state for UI-specific data only:

```python
# Navigation state
if "view" not in st.session_state:
    st.session_state.view = "welcome"

# Training ownership tracking
if "session_started_by_app" not in st.session_state:
    st.session_state.session_started_by_app = False
```

#### Persistent State
Critical application state persisted through service layer:
- Training configuration stored in ProcessManager
- Training status in file system
- Model checkpoints in designated folders

### Error Handling

#### User-Friendly Error Display
```python
# Consistent error patterns
try:
    result = training_service.start_training(config)
    st.success("Training started successfully!")
except Exception as e:
    st.error(f"Failed to start training: {e}")
```

#### Graceful Degradation
- Components continue functioning when data unavailable
- Clear status messages for all error states
- Recovery options provided for each error scenario

### Performance Optimization

#### Data Caching
- TensorBoard data cached with time-based invalidation
- Expensive computations cached and reused
- Smart cache clearing on training state changes

#### UI Responsiveness
- Fragment updates limited to 1-second intervals
- Conditional updates only when data changes
- Efficient data loading and processing

## Data Pipeline Support

### Supported Data Sources

#### 1. HuggingFace Datasets
- **Integration**: Direct HF Hub access
- **Authentication**: Token-based authentication support
- **Preprocessing**: Automatic data formatting and tokenization
- **Validation**: Real-time data validation and preview

#### 2. TensorFlow Datasets
- **TFDS Integration**: Native TensorFlow dataset support
- **Format Support**: Multiple data formats and structures
- **Preprocessing**: Automatic data pipeline creation
- **Optimization**: Efficient data loading and batching

#### 3. Custom JSON Upload
- **Format Flexibility**: Support for various JSON structures
- **Validation**: Real-time format validation
- **Preview**: Data preview before training
- **Error Handling**: Clear error messages for invalid data

### Data Processing Pipeline

```python
# Automatic data pipeline creation
pipeline = create_pipeline(data_config)
train_ds = pipeline.get_train_dataset()
```

**Features**:
- **Automatic Preprocessing**: Intelligent data preprocessing
- **Format Detection**: Automatic format detection and handling
- **Train/Val Split**: Configurable data splitting
- **Tokenization**: Model-appropriate tokenization

## Configuration Management

### Centralized Configuration
**Location**: `config/app_config.py`

**Configuration Categories**:
- **Model Options**: Available Gemma variants and capabilities
- **Training Methods**: Standard, LoRA, DPO configurations
- **System Settings**: File paths, ports, resource limits
- **Default Values**: Sensible defaults for all parameters

### Runtime Configuration
- **User Selections**: Stored in service layer during training
- **Parameter Validation**: Real-time validation and error checking
- **Configuration Persistence**: Maintained across browser refreshes

## User Experience Features

### Progressive Workflow
1. **Guided Setup**: Step-by-step model configuration
2. **Real-time Validation**: Immediate feedback on configuration
3. **Preview and Review**: Complete configuration review before training
4. **Live Monitoring**: Real-time training progress and metrics
5. **Interactive Testing**: Post-training model evaluation

### Responsive Design
- **Wide Layout**: Optimized for dashboard monitoring
- **Sidebar Navigation**: Quick access to all features
- **Mobile Compatibility**: Responsive design for various screen sizes
- **Professional Appearance**: Clean, modern interface design

### Error Recovery
- **Session Recovery**: Handle interrupted browser sessions
- **Process Recovery**: Detect and manage orphaned training processes
- **State Reset**: Clean application reset when needed
- **Clear Messaging**: Intuitive error messages and recovery steps

## Usage Patterns

### Typical Workflow
1. **Start**: Welcome page → Create Model
2. **Configure**: Select model → Choose data → Set parameters
3. **Review**: Configuration summary → Start training
4. **Monitor**: Real-time dashboard → Track progress
5. **Complete**: Training ends → Test model → Create new model

### Advanced Features
- **Checkpoint Management**: Automatic model checkpointing
- **Resource Monitoring**: System resource tracking
- **Error Analysis**: Comprehensive error detection and reporting
- **Performance Optimization**: Automatic performance tuning

## Future Enhancements

### Planned Improvements
1. **Multi-Model Training**: Support for concurrent training sessions
2. **Advanced Visualization**: Enhanced plotting and analytics
3. **Model Comparison**: Side-by-side model performance comparison
4. **Export Capabilities**: Model export and deployment features
5. **Collaboration Features**: Multi-user support and sharing

### Technical Improvements
1. **Performance Optimization**: Reduced memory usage and faster updates
2. **Enhanced Testing**: Comprehensive test coverage
3. **Better Error Handling**: More robust error recovery
4. **Documentation**: Interactive help and tutorials

This documentation reflects the current state of the UI after the service layer refactoring and provides a comprehensive guide for understanding and extending the interface.
