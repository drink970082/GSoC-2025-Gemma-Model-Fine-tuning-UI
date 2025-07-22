MODEL_INFO = {
    "Gemma2_2B": {
        "size": "2 billion parameters",
        "description": "A lightweight and efficient model suitable for tasks on devices with limited computational power, including high-end CPUs and consumer GPUs. It is designed for entry-level text generation, summarization, and simple Q&A.",
        "use_cases": [
            "On-device applications",
            "Simple chatbots and conversational AI",
            "Text summarization and generation for less complex topics",
            "Educational tools and quick experiments",
        ],
        "requirements": {
            "min_gpu_memory": "Approx. 12.7 GB of system RAM for CPU execution, or a GPU with at least 4 GB of VRAM for quantized versions (e.g., INT4).",
            "recommended_gpu": "NVIDIA RTX 3060 (12GB) or equivalent for better performance.",
        },
    },
    "Gemma2_9B": {
        "size": "9 billion parameters",
        "description": "A model that offers a significant performance improvement over the 2B version, suitable for a wider range of more complex tasks. It provides a good balance between performance and resource requirements.",
        "use_cases": [
            "Advanced text generation and summarization",
            "Content creation and assistance with writing",
            "More sophisticated chatbots and conversational agents",
            "Language understanding and reasoning tasks",
        ],
        "requirements": {
            "min_gpu_memory": "A GPU with at least 24 GB of VRAM is recommended for optimal performance.",
            "recommended_gpu": "NVIDIA RTX 3090 (24GB) or NVIDIA A100 (40GB).",
        },
    },
    "Gemma2_27B": {
        "size": "27 billion parameters",
        "description": "A powerful model that delivers high performance on a wide array of language tasks, rivaling larger models. It is capable of running on a single high-end GPU or TPU, making it accessible for demanding applications.",
        "use_cases": [
            "Complex reasoning and analysis",
            "In-depth document summarization and Q&A",
            "High-quality content and code generation",
            "Multilingual tasks and advanced research applications",
        ],
        "requirements": {
            "min_gpu_memory": "Can run on a single NVIDIA H100 GPU. For other setups, a GPU with 40GB+ of VRAM is ideal.",
            "recommended_gpu": "NVIDIA H100 (80GB) or NVIDIA A100 (40GB). Can also be run on CPU with at least 32GB of RAM for specific use cases.",
        },
    },
    "Gemma3_1B": {
        "size": "1 billion parameters",
        "description": "A highly efficient, text-only model designed for on-device applications where memory and processing power are very constrained. It has a 32K context window, making it suitable for lightweight tasks.",
        "use_cases": [
            "Mobile and edge device applications",
            "Basic text generation and simple chat functions",
            "Applications requiring a very small memory footprint (less than 1GB when quantized)",
        ],
        "requirements": {
            "min_gpu_memory": "Approximately 4 GB for full precision, and as low as 861 MB for 4-bit quantization.",
            "recommended_gpu": "NVIDIA GTX 1650 (4GB) or equivalent.",
        },
    },
    "Gemma3_4B": {
        "size": "4 billion parameters",
        "description": "A versatile multimodal model that balances strong performance with moderate resource requirements. It supports both text and image inputs and features a 128K context window.",
        "use_cases": [
            "Multimodal applications (text and image understanding)",
            "Advanced chatbots and content creation tools",
            "Analysis of visual data combined with text",
            "Applications requiring a balance of capability and efficiency",
        ],
        "requirements": {
            "min_gpu_memory": "Around 16 GB for full precision, and approximately 3.2 GB for 4-bit quantization.",
            "recommended_gpu": "NVIDIA RTX 3060 (12GB) for general use.",
        },
    },
    "Gemma3_12B": {
        "size": "12 billion parameters",
        "description": "A powerful multimodal model designed for more complex and demanding tasks. It offers enhanced language and vision capabilities with a 128K context window.",
        "use_cases": [
            "Complex, multi-turn conversational AI with image input",
            "In-depth analysis and reasoning over text and images",
            "High-quality content generation for professional applications",
            "Research and development in multimodal AI",
        ],
        "requirements": {
            "min_gpu_memory": "Approximately 48 GB for full precision, and about 8.2 GB for 4-bit quantization.",
            "recommended_gpu": "NVIDIA RTX 5090 (32GB) or NVIDIA A100 (40GB).",
        },
    },
    "Gemma3_27B": {
        "size": "27 billion parameters",
        "description": "The flagship multimodal model of the Gemma 3 family, providing state-of-the-art performance for the most demanding applications. It supports a 128K context window and advanced multimodal reasoning.",
        "use_cases": [
            "Enterprise-level AI applications",
            "Advanced research in natural language and vision processing",
            "Complex problem-solving and multi-document analysis with images",
            "Powering sophisticated AI agents and systems",
        ],
        "requirements": {
            "min_gpu_memory": "Roughly 108 GB for full precision, and about 19.9 GB for 4-bit quantization.",
            "recommended_gpu": "Multiple high-end GPUs like the NVIDIA RTX 4090 (24GB) or a single NVIDIA H100 (80GB).",
        },
    },
}
