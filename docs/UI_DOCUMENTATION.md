# Gemma Fine-tuning UI Documentation

## Overview

The Gemma Fine-tuning UI is a user-friendly interface for fine-tuning Google's Gemma models. It provides a streamlined workflow for creating, training, and testing fine-tuned models.

## Features

### 1. Create Model

- **Model Configuration**
  - Model name input
  - Base model selection
  - Training parameters configuration
  - Dataset selection and configuration

### 2. Training

- **Real-time Training Progress**

  - Progress bar showing overall training progress
  - Current epoch and loss display
  - Learning rate monitoring
  - GPU memory usage tracking
- **Training Metrics Visualization**

  - Interactive loss curve
  - Learning rate curve
  - Epoch markers
  - Real-time updates
- **Training Controls**

  - Start/Stop training
  - Abort training option
  - Training status indicators

### 3. Results

- **Model Playground**

  - Test fine-tuned model with custom inputs
  - View model responses
  - Compare with base model performance
- **Training History**

  - Final training metrics
  - Training duration
  - Resource utilization summary
  - Training plots and visualizations

## Data Pipeline Support

### Supported Data Sources

1. **JSON Format**

   - Structured text data
   - Optional labels
   - Automatic train/validation split
2. **HuggingFace Datasets**

   - Direct integration with HuggingFace
   - Support for various dataset formats
   - Automatic data preprocessing
3. **TensorFlow Datasets**

   - Support for TFDS datasets
   - Sequence-to-sequence task support
   - Custom tokenization

## Technical Details

### UI Components

- Built with Streamlit
- Real-time updates using session state
- Interactive visualizations with Plotly
- Responsive layout

### Data Processing

- Automatic data validation
- Train/validation split
- Tokenization handling
- Batch processing support

### Model Management

- Model configuration persistence
- Training state management
- Resource monitoring
- Model saving and loading

## Usage Guide

### Starting the UI

```bash
streamlit run app/Home.py
```

### Basic Workflow

1. Create a new model

   - Enter model name
   - Select base model
   - Configure training parameters
   - Choose dataset
2. Start Training

   - Monitor progress
   - View metrics
   - Stop if needed
3. Test Results

   - Use model playground (inferencing)

## Configuration

### Training Parameters

- Epochs
- Batch size
- Learning rate
- Validation split

### Data Pipeline Options

- Data source selection
- Preprocessing options
- Tokenization settings
- Batch size configuration

## Future Enhancements

- Additional data source support
- Advanced model configuration
- Enhanced visualization options
- Performance optimization
- Multi-model comparison
- Export/Import functionality
