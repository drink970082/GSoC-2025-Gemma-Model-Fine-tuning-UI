# Gemma Fine-Tuning UI

A modular interface for fine-tuning Google DeepMind’s Gemma language models. Supports real-time monitoring, multi-source data pipelines, and both standard and LoRA fine-tuning.

---

## Features

- **Frontend:** Streamlit UI, modular views (welcome, model creation, dashboard, inference), component-based.
- **Backend:** Python, modular. Core training, process management, data pipeline abstraction.
- **Data:** HuggingFace, TensorFlow, or JSON datasets. Tokenization and preview supported.
- **Training:** Standard and LoRA fine-tuning. Model variants loaded dynamically.
- **Monitoring:** Real-time metrics, logs, system resource usage, TensorBoard integration.
- **Service Layer:** Connects UI to backend, manages training lifecycle, metrics, logs.

---

## Project Structure

```
gemma_fine_tuning_ui/
├── app/                # Streamlit frontend
│   ├── components/     # UI components (inference, model creation, dashboard)
│   ├── view/           # Views (welcome, create, dashboard, inference)
│   └── main.py         # Streamlit entrypoint
├── backend/            # Backend logic
│   ├── core/           # Model, optimizer, loss, checkpoint, fine-tuner, trainer
│   ├── manager/        # Process, system, tensorboard, state management
│   ├── utils/          # Utilities
│   ├── data_pipeline.py# Data pipeline abstraction
│   └── trainer_main.py # Training entrypoint (CLI)
├── services/           # Service layer (DI, training service)
├── config/             # Config/dataclasses
├── requirements.txt    # Python dependencies
```

---

## Installation

**Prerequisites:**  
- Python 3.8–3.12  
- CUDA-compatible GPU recommended  
- Linux (WSL2 or native)

**Install:**
```bash
git clone https://github.com/drink970082/GSoC-2025-Gemma-Model-Fine-tuning-UI.git
cd gemma_fine_tuning_ui

# System deps
sudo apt-get update
sudo apt-get install -y pkg-config cmake

# Python deps
pip install -r requirements.txt

# JAX (choose one)
pip install jax           # CPU only
pip install jax[cuda12]   # NVIDIA GPU
```

---

## Usage

```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Workflow

1. **Welcome:** Choose to create a new model or use existing checkpoints.
2. **Model Creation:** Select model variant, fine-tuning method (Standard/LoRA), data source, and training params.
3. **Dashboard:** Monitor training progress, metrics, logs, and system usage.
4. **Inference:** Test your fine-tuned model interactively.

---

## Data Pipeline

- **Sources:** HuggingFace, TensorFlow, JSON
- **Preview:** Raw and tokenized preview before training
- **Transforms:** Configurable prompt/response fields, tokenization, truncation

---

## Training

- **Methods:** Standard, LoRA (DPO in config, not implemented)
- **Variants:** Any Gemma model supported by `gemma.gm.nn`
- **Checkpoints:** Automatic, with recovery and cleanup

---

## Monitoring

- **Metrics:** Real-time KPIs, loss curves, learning rates
- **System:** GPU/CPU/memory usage, logs, error reporting
- **TensorBoard:** Integrated

---

## Troubleshooting

- **Training stuck:** Check GPU, dataset, logs
- **UI unresponsive:** Refresh browser, restart Streamlit
- **OOM:** Lower batch size, use LoRA
- **Orphaned processes:** Use UI reset, check for stale lock files

---

## System Architecture
```mermaid
flowchart LR
    subgraph Frontend
        A1[Streamlit UI]
        A2["Views (welcome, create, dashboard, inference)"]
        A3["Components (model, data, dashboard, inference)"]
        A1 --> A2
        A2 --> A3
    end

    subgraph Service_Layer
        B1[TrainingService]
        B2[DI Container]
        B1 --> B2
    end

    subgraph Backend
        C1["Core (ModelTrainer, FineTuner)"]
        C2[DataPipeline]
        C3["Manager (Process, System, TensorBoard, State)"]
        C4[Utils]
        C1 --> C2
        C1 --> C3
        C3 --> C4
    end

    A1 -->|user actions, config| B1
    B1 -->|commands, config| C1
    C1 -->|metrics, logs| B1
    B1 -->|status, updates| A1
```

## Data Flow
```mermaid
sequenceDiagram
    participant User
    participant UI as Streamlit UI
    participant Service as TrainingService
    participant Backend as Backend Core

    User->>UI: Interact (create, train, monitor, infer)
    UI->>Service: Send config, commands
    Service->>Backend: Start training, fetch metrics/logs
    Backend-->>Service: Training status, metrics, logs
    Service-->>UI: Update dashboard, logs
    UI-->>User: Display results, progress
```

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Notes

- No support for Windows native.
- DPO/PEFT not implemented, only Standard and LoRA.
- For bug reports or feature requests, use GitHub Issues.
