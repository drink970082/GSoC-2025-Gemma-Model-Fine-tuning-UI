import sys

from backend.core.trainer import ModelTrainer
from backend.utils.cli import create_parser
from config.app_config import DataConfig, ModelConfig

# This print statement will be the very first thing to run.
print("trainer_main.py: Script execution started.")


def main():
    """Main entry point for the training script."""
    print("trainer_main.py: main() function entered.")
    parser = create_parser()
    args = parser.parse_args()
    print("trainer_main.py: Arguments parsed successfully.")

    try:
        data_config = DataConfig(**args.data_config)
        model_config = ModelConfig(**args.model_config)
        trainer = ModelTrainer(data_config, model_config)
        trainer.train()
    except Exception as e:
        print(
            f"trainer_main.py: An unhandled exception occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
