FINE_TUNING_METHODS = {
    "Standard": {
        "description": "Full fine-tuning of all model parameters for maximum adaptation and performance.",
        "advantages": [
            "Best possible accuracy",
            "Complete control over model behavior",
        ],
        "disadvantages": [
            "High GPU memory and compute requirements",
            "Slower training",
        ],
        "best_for": [
            "When you need the highest accuracy",
            "When resources are not a constraint",
        ],
        "memory_usage": "High",
        "training_speed": "Fast",
        "use_case": "Full adaptation",
        "default_parameters": {
            "epochs": 300,
            "learning_rate": 1e-3,
        },
    },
    "LoRA": {
        "description": "Parameter-efficient fine-tuning using low-rank adapters for faster, lighter training.",
        "parameters": {
            "lora_rank": {
                "description": "Rank of the LoRA update matrices",
                "range": "1-32",
                "default": 4,
                "effect": "Higher rank = more capacity, more parameters",
            },
        },
        "advantages": [
            "Much lower memory usage",
            "Faster and cheaper to train",
            "Easy to store multiple fine-tuned variants",
        ],
        "disadvantages": [
            "Slightly less accurate than full fine-tuning",
            "Limited flexibility",
        ],
        "best_for": [
            "Limited GPU memory",
            "Rapid prototyping",
        ],
        "memory_usage": "Low",
        "training_speed": "Medium",
        "use_case": "Parameter-efficient",
        "default_parameters": {
            "epochs": 500,
            "learning_rate": 5e-3,
        },
    },
    "QuantizationAware": {
        "description": "Trains the model to be robust to quantization, enabling efficient deployment on edge devices.",
        "advantages": [
            "Smallest model size",
            "Fastest inference",
            "Best for mobile/edge deployment",
        ],
        "disadvantages": [
            "Slightly more complex training",
            "May lose some accuracy",
        ],
        "best_for": [
            "Deploying to resource-constrained hardware",
            "When model size and speed are critical",
        ],
        "memory_usage": "Very Low",
        "training_speed": "Slow",
        "use_case": "Resource-constrained",
        "default_parameters": {
            "epochs": 1000,
            "learning_rate": 5e-3,
        },
    },
}
