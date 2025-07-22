import argparse
import json
from typing import Any, Dict


def parse_json_arg(arg: str) -> Dict[str, Any]:
    """Parse a JSON argument from command line."""
    try:
        return json.loads(arg)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format: {e}")


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=parse_json_arg, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    return parser
