# tests/backend/utils/test_cli.py
import argparse
import json

import pytest

from backend.utils.cli import create_parser, parse_json_arg


def test_parse_json_arg_valid():
    arg = json.dumps({"a": 1, "b": [1, 2]})
    out = parse_json_arg(arg)
    assert out == {"a": 1, "b": [1, 2]}


def test_parse_json_arg_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_json_arg("{bad json]")


def test_create_parser_parses_required_args(tmp_path):
    parser = create_parser()
    cfg = {"x": 1}
    args = parser.parse_args(["--config", json.dumps(cfg), "--work_dir", str(tmp_path)])
    assert args.config == cfg
    assert args.work_dir == str(tmp_path)