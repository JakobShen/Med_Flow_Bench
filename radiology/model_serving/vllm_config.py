#!/usr/bin/env python3
"""Read and validate the Table 1 model manifest for shell launchers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "shared" / "config" / "table1_models.json"
REQUIRED_MODEL_FIELDS = {"paper_label", "category", "runtime", "provider", "model"}


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported schema_version in {path}")
    models = payload.get("models")
    sets = payload.get("sets")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{path} must define a non-empty models object")
    if not isinstance(sets, dict) or not sets:
        raise ValueError(f"{path} must define a non-empty sets object")

    for key, record in models.items():
        if not isinstance(record, dict):
            raise ValueError(f"Model {key!r} must be an object")
        missing = REQUIRED_MODEL_FIELDS - set(record)
        if missing:
            raise ValueError(f"Model {key!r} is missing: {', '.join(sorted(missing))}")
        runtime = record["runtime"]
        if runtime not in {"hosted", "vllm"}:
            raise ValueError(f"Model {key!r} has unsupported runtime {runtime!r}")
        if runtime == "vllm":
            for field in ("source_model", "served_model", "vllm_args"):
                if field not in record:
                    raise ValueError(f"vLLM model {key!r} is missing {field!r}")
            if not isinstance(record["vllm_args"], list) or not all(
                isinstance(value, str) and "\n" not in value for value in record["vllm_args"]
            ):
                raise ValueError(f"vLLM model {key!r} has invalid vllm_args")

    for set_name, keys in sets.items():
        if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
            raise ValueError(f"Model set {set_name!r} must be a string list")
        unknown = [key for key in keys if key not in models]
        if unknown:
            raise ValueError(f"Model set {set_name!r} references unknown models: {unknown}")
    return payload


def get_model(config: dict[str, Any], key: str) -> dict[str, Any]:
    try:
        return config["models"][key]
    except KeyError as exc:
        choices = ", ".join(config["models"])
        raise ValueError(f"Unknown model key {key!r}. Available: {choices}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Print one model key per line")
    list_parser.add_argument("--set", dest="set_name", default="all")

    get_parser = subparsers.add_parser("get", help="Print one scalar model field")
    get_parser.add_argument("model_key")
    get_parser.add_argument("field")

    args_parser = subparsers.add_parser("serve-args", help="Print one vLLM argument per line")
    args_parser.add_argument("model_key")

    subparsers.add_parser("validate", help="Validate the manifest")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config.expanduser().resolve())
        if args.command == "validate":
            print(f"OK {args.config}")
            return 0
        if args.command == "list":
            try:
                keys = config["sets"][args.set_name]
            except KeyError as exc:
                choices = ", ".join(config["sets"])
                raise ValueError(f"Unknown model set {args.set_name!r}. Available: {choices}") from exc
            print("\n".join(keys))
            return 0

        model = get_model(config, args.model_key)
        if args.command == "serve-args":
            if model["runtime"] != "vllm":
                raise ValueError(f"{args.model_key!r} is not a vLLM model")
            print("\n".join(model["vllm_args"]))
            return 0

        value = model.get(args.field)
        if value is None or isinstance(value, (dict, list)):
            raise ValueError(f"{args.model_key!r}.{args.field} is not a scalar field")
        if isinstance(value, bool):
            print(str(value).lower())
        else:
            print(value)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"vllm_config: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
