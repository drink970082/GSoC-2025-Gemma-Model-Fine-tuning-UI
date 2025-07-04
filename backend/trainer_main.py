import sys

from backend.core.trainer import ModelTrainer
from backend.utils.cli import create_parser
from config.dataclass import (
    TrainingConfig,
    ModelConfig,
    DataConfig,
    MethodConfig,
    LoraParams,
    DpoParams,
)


def parse_config(config: dict) -> TrainingConfig:
    # Handle the parameters field
    parameters = None
    method_config_dict = config["method_config"]
    if method_config_dict.get("parameters"):
        if method_config_dict["name"] == "LoRA":
            parameters = LoraParams(**method_config_dict["parameters"])
        elif method_config_dict["name"] == "DPO":
            parameters = DpoParams(**method_config_dict["parameters"])
    method_config = MethodConfig(
        name=method_config_dict["name"], parameters=parameters
    )
    # Standard method has parameters=None, which is already set
    return TrainingConfig(
        model_name=config["model_name"],
        model_config=ModelConfig(**config["model_config"]),
        data_config=DataConfig(**config["data_config"]),
        method_config=method_config,
    )


# This print statement will be the very first thing to run.
print("trainer_main.py: Script execution started.")


def main():
    """Main entry point for the training script."""
    print("trainer_main.py: main() function entered.")
    parser = create_parser()
    args = parser.parse_args()
    print("trainer_main.py: Arguments parsed successfully.")

    try:
        training_config = parse_config(args.config)
        work_dir = args.work_dir
        trainer = ModelTrainer(training_config, work_dir)
        trainer.train()
    except Exception as e:
        print(
            f"trainer_main.py: An unhandled exception occurred: {e}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        print("trainer_main.py: Script execution completed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
