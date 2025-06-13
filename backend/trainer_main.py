import sys
from backend.utils.cli import create_parser
from config.training_config import ModelConfig, DataConfig
from backend.core.trainer import ModelTrainer


def main():
    """Main entry point for the training script."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        data_config = DataConfig(**args.data_config)
        model_config = ModelConfig(**args.model_config)
        trainer = ModelTrainer(data_config, model_config)
        trainer.train()
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
