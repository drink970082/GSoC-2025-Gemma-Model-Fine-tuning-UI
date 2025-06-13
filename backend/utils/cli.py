import argparse
import ast
from typing import Dict


def parse_dict_arg(arg: str) -> Dict:
    """Parse a dictionary argument from command line."""
    try:
        return ast.literal_eval(arg)
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=parse_dict_arg, required=True)
    parser.add_argument("--model_config", type=parse_dict_arg, required=True)
    return parser
