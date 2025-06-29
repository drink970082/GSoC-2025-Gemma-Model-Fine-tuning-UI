# Gemma Fine-Tuning UI

A comprehensive web interface for fine-tuning Google DeepMind's Gemma language models with real-time monitoring, intuitive workflow management, and powerful inference capabilities.

## What This Project Does

The Gemma Fine-Tuning UI transforms the complex process of fine-tuning large language models into an accessible, visual workflow. It provides:

### Complete Training Pipeline
- **Model Configuration**: Choose from Gemma 2B, 9B, 27B, or Gemma 3 (1B, 4B, 12B, 27B) variants
- **Data Integration**: Support for HuggingFace datasets, TensorFlow datasets, and custom JSON uploads
- **Training Methods**: Standard fine-tuning
- **Real-time Monitoring**: Live tracking of training progress, metrics, and system resources

### Advanced Monitoring Dashboard
- **Live Metrics**: Real-time loss curves, learning rates, and performance indicators
- **System Monitoring**: GPU/CPU utilization, memory usage, and temperature tracking
- **Error Tracking**: Comprehensive logging with error detection and reporting

### Interactive Inference Playground
- **Model Testing**: Test your fine-tuned models with custom prompts

### Robust Process Management
- **Training Control**: Start, stop, abort, and resume training sessions
- **Session Recovery**: Handle interrupted sessions and orphaned processes
- **Resource Cleanup**: Automatic cleanup of temporary files and processes
- **State Persistence**: Maintain training state across browser refreshes

## Architecture Overview

### Service-Oriented Design
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI Layer      │    │  Service Layer  │    │ Backend Layer   │
│                 │    │                 │    │                 │
│ • Views         │◄──►│ • TrainingService│◄──►│ • ProcessManager│
│ • Components    │    │ • StateManager  │    │ • CoreTrainer   │
│ • Controls      │    │ • ErrorHandler  │    │ • DataPipeline  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

**Frontend (Streamlit)**
- `app/view/` - Main application views (welcome, create, dashboard, inference)
- `app/components/` - Reusable UI components organized by feature
- `app/services/` - Business logic abstraction layer

**Backend (Python)**
- `backend/core/` - Training engine, model management, and data processing
- `backend/manager/` - Process, TensorBoard, and status management
- `config/` - Centralized configuration management

## 📋 Features in Detail

### 1. Model Creation Workflow
- **Step-by-step Configuration**: Guided process for setting up training
- **Parameter Validation**: Real-time validation of training parameters
- **Data Preview**: Preview and validate your training data before starting
- **Configuration Summary**: Review all settings before initiating training

### 2. Training Dashboard
- **Live Progress Tracking**: Real-time updates on training progress
- **Performance Metrics**: KPIs including training speed, ETA, and resource utilization
- **Interactive Charts**: Dynamic plots for loss curves and learning rates
- **System Resource Monitoring**: Live tracking of GPU/CPU usage and memory

### 3. Data Pipeline Support
- **Multiple Sources**: HuggingFace Hub, TensorFlow Datasets, custom JSON files
- **Automatic Preprocessing**: Intelligent data preprocessing and tokenization
- **Format Detection**: Automatic detection and handling of different data formats
- **Validation**: Built-in data validation and error reporting

### 4. Advanced Training Features
- **Fine-tuning Methods**:
  - **Standard**: Full parameter fine-tuning for maximum performance
- **Flexible Configuration**: Customizable epochs, batch size, learning rate
- **Checkpoint Management**: Automatic model checkpointing and recovery

### 5. Real-time Monitoring
- **TensorBoard Integration**: Full TensorBoard embedded in the interface
- **Live Metrics**: Real-time tracking of training and performance metrics
- **Error Detection**: Automatic error detection with detailed reporting
- **Resource Monitoring**: System resource usage with historical tracking

## Getting Started

### Prerequisites
- **Python 3.8-3.12**
- **CUDA-compatible GPU** (recommended for training)
- **8GB+ RAM** (16GB+ recommended)

### Quick Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/drink970082/GSoC-2025-Gemma-Model-Fine-tuning-UI.git
   cd gemma-fine-tuning-ui
   ```

2. Install JAX (choose your platform):
   ```bash
   # For CPU only
   pip install jax

   # For NVIDIA GPU
   pip install jax[cuda12]

   # For TPU
   pip install jax[tpu]
   ```

3. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y pkg-config cmake
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Launch the application:
   ```bash
   streamlit run app/main.py
   ```

6. Open your browser and navigate to `http://localhost:8501`

### First Run Workflow

1. Start on Welcome Page: Choose to create a new model or use existing checkpoints
2. Configure Your Model: 
   - Enter a model name
   - Select fine-tuning method (Standard)
   - Choose your data source
   - Select Gemma model variant
   - Set training parameters
3. Review Configuration: Preview all settings before starting
4. Start Training: Monitor progress in real-time dashboard
5. Test Your Model: Use the inference playground to test results

## Project Structure

```
gemma-fine-tuning-ui/
├── app/                          # Frontend application
│   ├── components/              # Reusable UI components
│   │   ├── create_model/       # Model creation workflow
│   │   ├── training_dashboard/ # Training monitoring
│   │   └── inference/          # Model testing
│   ├── services/               # Business logic layer
│   ├── view/                   # Main application views
│   └── main.py                 # Application entry point
├── backend/                     # Backend functionality
│   ├── core/                   # Training engine
│   ├── manager/                # Process management
│   └── utils/                  # Utility functions
├── config/                     # Configuration management
├── docs/                       # Documentation
├── tests/                      # Test suite
└── requirements.txt            # Dependencies
```

## Configuration

### Model Options
- **Gemma 2**: 2B, 9B, 27B parameters
- **Gemma 3**: 1B, 4B, 12B, 27B parameters (multimodal support)

### Training Methods
- **Standard Fine-tuning**: Full parameter training

### Data Sources
- **HuggingFace Datasets**: Direct integration with HF Hub
- **TensorFlow Datasets**: TFDS compatibility
- **Custom JSON**: Upload your own training data

## Development

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Troubleshooting

### Common Issues

**Training won't start:**
- Check GPU availability and memory
- Verify dataset accessibility
- Review error logs in the dashboard

**UI not responding:**
- Refresh the browser
- Check console for JavaScript errors
- Restart the Streamlit server

**Out of memory errors:**
- Reduce batch size
- Use LoRA instead of standard fine-tuning
- Monitor system resources

**Orphaned training processes:**
- Use the "Reset Application" button
- Check for stale lock files
- Restart the application

## Contributing

We welcome contributions! Please see our contributing guidelines and feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Google DeepMind** for the Gemma models
- **Google Summer of Code 2025** for project support

---

**Made with ❤️ for the open source community**
