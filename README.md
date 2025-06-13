# Gemma Model Fine-tuning UI

A user-friendly web interface for fine-tuning Google DeepMind's Gemma models, developed as part of Google Summer of Code 2025.

## Features

- Interactive data exploration and visualization
- Streamlined model fine-tuning workflow
- Real-time training monitoring
- Model evaluation and inferencing tools
- Export and deployment capabilities

## Prerequisites

- Python 3.8+ (Python 3.13 doesn't support kauldron)
- CUDA-compatible GPU (recommended)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/drink970082/GSoC-2025-Gemma-Model-Fine-tuning-UI.git
cd gemma-fine-tuning-ui
```

2. Install JAX for CPU, GPU or TPU. Follow the instructions on [the JAX website](https://jax.readthedocs.io/en/latest/installation.html).
3. Install dependencies:

   ```bash
   # Install cmake and pkg-config for Gemma module
   sudo apt-get update
   sudo apt-get install -y pkg-config cmake
   # Project Dependencies
   pip install requirements.txt
   ```
4. To utilize GPU training, you may need to install cuda toolkit and cuDNN. Follow the instructions on [Tensorflow website](https://www.tensorflow.org/install/gpu)

## Usage

1. Start the application:

```bash
streamlit run app/Home.py
```

2. Open your browser and navigate to `http://localhost:8501`
3. Follow the on-screen instructions to:

   - Upload your dataset
   - Configure model parameters
   - Start fine-tuning
   - Monitor training progress
   - Evaluate results

## Project Structure

```
gemma-fine-tuning-ui/
├── app/                # Streamlit application
├── backend/           # Core functionality
├── tests/            # Test suite
└── requirements.txt  # Python dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google DeepMind for the Gemma models
- Google Summer of Code 2025
- All contributors and supporters

## Contact

Project Link: [https://github.com/yourusername/gemma-fine-tuning-ui](https://github.com/yourusername/gemma-fine-tuning-ui)
