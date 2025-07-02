import argparse
import ast
from typing import Dict
import json


def parse_dict_arg(arg: str) -> Dict:
    """Parse a dictionary argument from command line."""
    try:
        return ast.literal_eval(arg)
    except (SyntaxError, ValueError) as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")


def parse_json_arg(arg: str) -> Dict:
    # The 'arg' parameter is the big JSON string from the command line.
    return json.loads(arg)


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=parse_json_arg, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    return parser
