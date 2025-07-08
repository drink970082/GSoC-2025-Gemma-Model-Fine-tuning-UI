FINE_TUNING_METHODS = {
    "Standard": {
        "description": "Full fine-tuning of all model parameters",
        "advantages": [
            "Best performance potential",
            "Complete model adaptation",
            "No architectural changes",
        ],
        "disadvantages": [
            "Requires more GPU memory",
            "Longer training time",
            "Higher computational cost",
        ],
        "best_for": [
            "When you have sufficient computational resources",
            "When maximum performance is required",
            "When you need to modify the entire model",
        ],
    },
    "LoRA": {
        "description": "Low-Rank Adaptation: Efficient fine-tuning method that adds small trainable rank decomposition matrices",
        "parameters": {
            "lora_rank": {
                "description": "Rank of the LoRA update matrices",
                "range": "1-32",
                "default": 4,
                "effect": "Higher rank = more capacity but more parameters",
            },
        },
        "advantages": [
            "Much lower memory usage",
            "Faster training",
            "Can be merged with base model",
            "Parameter efficient",
        ],
        "disadvantages": [
            "Slightly lower performance than full fine-tuning",
            "Limited adaptation capacity",
        ],
        "best_for": [
            "Limited GPU memory",
            "Quick experiments",
            "When you need to save multiple fine-tuned versions",
        ],
    },
    "DPO": {
        "description": "Direct Preference Optimization: Fine-tunes model based on human preferences",
        "parameters": {
            "dpo_beta": {
                "description": "Temperature parameter for DPO",
                "range": "0.1-1.0",
                "default": 0.1,
                "effect": "Higher beta = more exploration",
            }
        },
        "advantages": [
            "Better alignment with human preferences",
            "Can improve model behavior",
            "More controlled outputs",
        ],
        "disadvantages": [
            "Requires preference data",
            "More complex training process",
            "May need more iterations",
        ],
        "best_for": [
            "When you have preference data",
            "When you need to align model with specific behaviors",
            "When you want to improve model outputs",
        ],
    },
}
