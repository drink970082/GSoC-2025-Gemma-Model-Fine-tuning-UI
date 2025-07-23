import sys

from backend.core.trainer import ModelTrainer
from backend.utils.cli import create_parser
from config.dataclass import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    LoraParams,
    DpoParams,
)


def parse_config(config: dict) -> TrainingConfig:
    # Handle the parameters field
    parameters = None
    model_config_dict = config["model_config"]
    if model_config_dict.get("parameters"):
        if model_config_dict["method"] == "LoRA":
            parameters = LoraParams(**model_config_dict["parameters"])
        elif model_config_dict["method"] == "DPO":
            parameters = DpoParams(**model_config_dict["parameters"])
    model_config = ModelConfig(
        model_variant=model_config_dict["model_variant"],
        epochs=model_config_dict["epochs"],
        learning_rate=model_config_dict["learning_rate"],
        method=model_config_dict["method"],
        parameters=parameters,
    )
    # Standard method has parameters=None, which is already set
    return TrainingConfig(
        model_name=config["model_name"],
        model_config=model_config,
        data_config=DataConfig(**config["data_config"]),
    )


def main() -> None:
    """Main entry point for the training script."""
    print("trainer_main.py: Script execution started.")

    try:
        parser = create_parser()
        args = parser.parse_args()
        print("trainer_main.py: Arguments parsed successfully.")

        training_config = parse_config(args.config)
        work_dir = args.work_dir
        trainer = ModelTrainer(training_config, work_dir)
        trainer.train()
        print("trainer_main.py: Script execution completed.")
        sys.exit(0)
    except ValueError as e:
        print(f"trainer_main.py: Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(
            f"trainer_main.py: An unhandled exception occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(1)



if __name__ == "__main__":
    main()
