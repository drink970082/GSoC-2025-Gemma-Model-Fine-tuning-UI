---
title: GSoC Proposal

---

# GSoC Proposal: Gemma Model Fine-tuning UI

## About Me

Hello, this is Howard from Taiwan! I am a first-year master's student at Duke University majoring in Financial Technology. I've had various projects and work experiences related to machine learning and full-stack development, and  I hope to build a simple, interactive tool that will make machine learning available to everyone.

When using Tensorboard or experimenting in Google AI Studio, I've found it helpful to visually monitor model performance, such as tracking loss/reward curves, retrieving evaluation metrics in real time, and experimenting with different hyperparameters. These interfaces not only make the training process smoother but also help with debugging, hyperparameter tuning, and understanding model behavior. I believe that building well-designed interfaces can make machine learning more intuitive and effective.

On the machine learning side, I worked with TensorFlow during a research assistantship focused on cross-domain recommendation systems. Most of my course projects and an internship at [Yunata Securities](https://www.yuanta.com.tw/eyuanta/securities/en/index) involved PyTorch. These covered a comprehensive range of tasks, including dataset preprocessing, various [deep learning](https://github.com/drink970082/CoachAI-Projects) and [reinforcement learning](https://github.com/drink970082/Selected-Topics-in-Reinforcement-Learning) model training, and performance evaluation.

Beyond ML, I’ve actively developed my web development and full-stack engineering skills. I’ve built projects using [HTML/CSS with Bootstrap](https://github.com/drink970082/Win88-Sport-Betting), [Node.js with Express](https://github.com/drink970082/Yelp-Camp), and Python frameworks like Flask and FastAPI. On the backend, I’ve worked with relational databases (MySQL, PostgreSQL), NoSQL databases (MongoDB), graph databases (Neo4j), and vector databases ([ChromaDB](https://github.com/drink970082/SustAInData)). I’m currently working with classmates on a [blockchain crowdfunding platform](https://ledgervest.colab.duke.edu/), where I’m responsible for frontend and backend integration.

Regarding cloud platforms, I recently passed the [AWS AI Practitioner Certification](https://www.credly.com/badges/bb6fd5d9-b09b-4c3e-aae7-beb1dc4918f7/public_url) and have used Google Cloud Platform to deploy various projects — including a Minecraft server and widgets for my algorithmic trading system.

You can learn more about me by checking out my [GitHub](https://github.com/drink970082)
Contact Information:
* Name: Chen-Hao Wu (you can also call me Howard)
* Email: howdywu@gmail.com
* LinkedIn: https://www.linkedin.com/in/chen-hao-wu-673314224/
* Location: Durham, NC, USA (UTC -4 (EDT))
* Language: English, Chinese 

## Project Description - Gemma Model Fine-tuning UI

### Abstract

Gemma is a lightweight, open-source large language model by Google DeepMind. This project aims to build an intuitive web interface for fine-tuning Gemma models. The interface will allow users to upload datasets, configure hyperparameters, monitor training progress, and export trained models — all without writing a single line of code. By lowering the entry barrier, the UI will empower a broader range of users to experiment with and adapt large language models to their specific tasks.

### Motivation

[Current implementation](https://github.com/google-deepmind/gemma/blob/main/colabs/finetuning.ipynb) of fine-tuning Gemma requires Python environment installation and knowledge of machine learning frameworks. This makes it difficult for people without coding experience to get started. This project aims to bridge this gap by wrapping the fine-tuning process in an interactive, no-code web interface, simplifying the setup process and experimentation with model settings. The UI will make fine-tuning accessible to a broader group of people, including teachers, researchers, and non-developers.

Personally, while learning through the Gemini Learning Path on Google Cloud, I enjoyed experimenting with hyperparameters like temperature or Top-P, and fine-tuning models using different datasets in Google AI Studio, and this experience inspired me to build a tool that brings similar interactivity and ease of use to the Gemma ecosystem. This project excites me not only technologically but also because it aligns with my passion for developing products combining machine learning with user experience. I believe contributing to this open-source project will be an invaluable opportunity to grow as both a developer and a member of the broader AI community.

### Proposed Deliverables

1. Interactive Fine-Tuning Web UI: A fully functional web interface (likely built with Streamlit or similar frameworks) that allows users to:
    * Upload and preprocess custom datasets
    * Configure and tune key hyperparameters
    * Visualize training progress (loss curves, evaluation metrics)
    * Load checkpoints and resume training
    * Export trained models and weights in various formats
    
2. Documentation and Usage Examples
    * Setup instructions for running the UI locally or on the cloud
    * Tutorials or example workflows demonstrating how to fine-tune Gemma models using the interface
    * Descriptions of available parameters and features

3. (Optional) Integration with Google Cloud Storage and Vertex AI for scalable training

### Benefits to the Community
This project will significantly lower the barrier to entry for working with Gemma, an open-source large language model. Currently, the fine-tuning process is limited to users with technical backgrounds who are comfortable working in code notebooks. 

By introducing a no-code web interface, this project will make Gemma’s powerful capabilities more accessible to a broader audience.
This tool will be especially valuable for:
* Educators and students who can use the UI for hands-on learning without needing to dive into low-level code
* Researchers and domain experts who may want to experiment with LLMs on their custom datasets without managing infrastructure or writing scripts
* Prototypers and ML practitioners who want a quick and visual way to test ideas, iterate on models, and evaluate results


### Project Breakdown

This section outlines the core components of the proposed Gemma Model Fine-Tuning UI. The goal is to streamline the fine-tuning process through an intuitive interface that abstracts away complexity while maintaining flexibility and transparency for advanced users.

The demo pages presented in the following sections serve as visual prototypes of a potential interface design, implemented using Streamlit.

#### Data Pipeline

This module enables users to upload, validate, preprocess, and augment datasets for training. It supports various formats and simplifies data handling for non-expert users.
* Uploading: Supports data ingestion from [kauldron](https://kauldron.readthedocs.io/en/latest/index.html)-compatible formats, enabling seamless integration with widely-used dataset structures
* Validation: Performs automatic checks to ensure data integrity and compatibility (e.g., input shapes, missing values, supported data types)
* Preprocessing: Provides common preprocessing operations such as reshaping, normalization, and tokenization, with customizable settings
* Data Augmentation: Offers optional techniques such as prompt variation, noise injection, and synthetic data generation to improve model generalizability

##### Demo page

![Figure 1: Upload and Explore Dataset](https://hackmd.io/_uploads/ryJJzmo6ke.png)
* Upload section: Allows users to upload custom datasets directly from their local machine, with potential support for integration with Hugging Face datasets, Google Drive, or other data providers
* Dataset Preview: Displays the first five rows of the uploaded dataset
* Input/Output Columns Selection: Allows users to designate specific columns as input features and/or target output labels
* Preprocessing Options: Provides a list of available preprocessing methods
* Data Summary: Presents a statistical overview and sample input/output of the preprocessed dataset


#### Hyperparameter Configuration

This module provides an intuitive interface for users to set and tune hyperparameters, even if they are new to machine learning.
* Interactive UI Components: Includes sliders, dropdowns, and number inputs with tooltips and built-in constraints to prevent invalid entries
* Default Profiles: Offers predefined hyperparameter configurations for common scenarios, allowing users to start quickly or run baseline experiments
* Helpful Tips: Displays inline documentation and best-practice suggestions based on the selected model or dataset type
* Checkpoint Support: Enables loading of model checkpoints to resume training or continue from a pre-trained state

##### Demo Page

![Figure 2: Configure Hyperparameters and Train](https://hackmd.io/_uploads/HyRiz7op1l.png)
* Learning Rate: Supports input in both standard and exponential notation (e.g., 1e-5)
* Batch Size and Epochs: Configurable via interactive sliders
* Tips: Provide contextual hints and explanations next to each hyperparameter field
* Default value: Displays recommended default settings for each hyperparameter, based on the selected model profile

#### Visualizing Training Progress

This module focuses on providing real-time feedback and analytics during training, inspired by tools like TensorBoard.
* Loss Curves: Displays real-time plots of training/validation loss across epochs, with options to overlay results from different runs for comparison
* Evaluation Metrics: Support standard metrics (e.g., Accuracy, ROC-AUC, F1-Score, Confusion Matrix), enabling visual comparison against baseline models or benchmark datasets
* Run History: Optionally caching past training sessions to support comparison, reproducibility, and experiment tracking

##### Demo Page

![Figure 3:Training Progress page with loss and accuracy curves, and sample outputs by epoch](https://hackmd.io/_uploads/B152zQsTkl.png)
* Training Status Hints: Notifies users of the current stage of the fine-tuning process
* Real-time loss curve: Displays a dynamically updating plot of the training loss/accuracy over time
* Sample Input/Output per Epoch: Shows model-generated predictions for selected sample inputs at each epoch

#### Model Download/Export Options

This module simplifies the process of exporting trained models for deployment or future reuse.
* Model Weights Export: Allows users to download trained model weights in standard formats such as PyTorch .pt or TensorFlow .h5
* Entire Model Serialization: Supports exporting the complete model, including architecture definitions and tokenizer configurations
* Integration-ready: Option to generate metadata or inference-ready code snippets for downstream integration (e.g., in REST APIs or Hugging Face Spaces).

##### Demo Page

![Figure 4: Summary and Model Export](https://hackmd.io/_uploads/S14kmXopJe.png)
* Export Model: Provides a user-friendly option to download the fine-tuned model
* Model Information: Displays a comprehensive summary of the training session, including base model used, total training time, hyperparameter settings, and final loss curves

#### Optional: Integration with Google Cloud Storage/Vertex AI
As an optional enhancement, the UI could support training and storage integration with Google Cloud services to enable large-scale fine-tuning and model management.

## Timeline

### 5/8~6/1 (Community Bonding Period)
Objectives:
* Familiarize myself with Kauldron and Gemma, the key backend and LLM packages for this project
* Explore Streamlit and assess its suitability for building the interactive UI
* Engage with the mentor and the Gemma community to clarify expectations, gather feedback, and refine the project scope
* Maintain regular communication with my mentor and stay active in community discussions
Activities:
* Conduct hands-on experiments with Gemma:
    * Fine-tune models using different datasets and configurations
    * Document key workflows and APIs to better understand the codebase and data pipeline
* Finalize design decisions for the UI framework and backend interaction structure

### 6/2~7/14 (Coding Phase I)
Goals:
* Develop a Minimum Viable Product (MVP) version of the UI

Deliverables:
* Dataset upload and preprocessing interface
* Basic model fine-tuning workflow (triggering training with default parameters)
* Model saving/exporting options

Additional Work:
* Begin initial testing and integration with Gemma backend

Detailed Timeline:
* Week 1: Implement file upload and dataset preview
* Week 2: Implement preprocessing options
* Week 3: Connect to Gemma fine-tuning Backend with default hyperparameters
* Week 4: Implement model saving
* Week 5-6: Buffer for testing and integration

### 7/14~7/18 (Midterm Evaluation)
Deliverables:
* Submit the midterm evaluation report
* Share mentor feedback and self-assessment
* Updated timeline and deliverables if adjustments are necessary
    
### 7/18~8/25 (Coding Phase II)
Goals:
* Extend and polish the UI

Planned Features:
* Add advanced features
    * data augmentation
    * real-time visualization of training progress
    * Support for checkpoint-based training.
* Improve the UX with tooltips, input validation, and default presets.
* If time and scope allow, explore integration with Google Vertex AI for seamless model deployment or training at scale.

Detailed Timeline:
* Week 8: Implement Data augmentation
* Week 9: Implement Real-time visualization
* Week 10: Implement checkpoint and UX improvement
* Week 11: Implement tooltips and default presets
* Week 12-13: Implement Vertex AI integration, documentation, and buffer for GitHub repository integration

### 8/25~9.1 (Final Week)
Wrap-Up Tasks:
* Finalize all components of the UI and conduct full testing
* Write comprehensive documentation for both users and contributors
* Prepare and submit the final project report and code submission

### Schedule Gantt Chart
![Gantt Chart](https://hackmd.io/_uploads/ry648VeR1x.png)

### Availability 

I'll be fully available throughout the GSoC period and have no academic or professional commitments during the summer. My summer break begins in early May and continues until the last week of August, during which I can dedicate 30–40 hours per week to the project.

Classes resume during the final week of GSoC, but I will ensure that all major components, documentation, and final reports are completed and submitted ahead of time.