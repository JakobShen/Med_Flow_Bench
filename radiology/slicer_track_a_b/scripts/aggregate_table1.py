#!/usr/bin/env python3
"""Aggregate Track A Paper_Table_Data.json files into Table 1 columns."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "shared" / "config" / "table1_models.json"
DATASETS = ("lumiere", "ucsf_pdgm", "nsclc")
CSV_COLUMNS = (
    "Model",
    "Category",
    "Longitudinal Task",
    "Longitudinal Strict",
    "Brain Task",
    "Brain Strict",
    "Brain Loc.",
    "Lung Task",
    "Lung Strict",
    "Lung Loc.",
    "Model Avg. Task",
    "Model Avg. Strict",
    "Model Avg. Loc.",
)


def newest_paper_file(directory: Path) -> Optional[Path]:
    candidates = list(directory.rglob("Paper_Table_Data.json")) if directory.exists() else []
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def as_number(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_present(values: list[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if len(present) == len(values) else None


def load_dataset_payload(model_dir: Path, dataset: str) -> tuple[dict[str, Any], Optional[Path]]:
    source = newest_paper_file(model_dir / dataset)
    if source is None:
        return {}, None
    return json.loads(source.read_text(encoding="utf-8")), source


def build_row(
    *,
    key: str,
    record: dict[str, Any],
    results_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_dir = results_root / key
    lumiere, lumiere_source = load_dataset_payload(model_dir, "lumiere")
    ucsf, ucsf_source = load_dataset_payload(model_dir, "ucsf_pdgm")
    nsclc, nsclc_source = load_dataset_payload(model_dir, "nsclc")

    long_task = as_number(lumiere.get("TaskAcc"))
    long_strict = as_number(lumiere.get("Strict"))
    brain_task = as_number(ucsf.get("Task"))
    brain_strict = as_number(ucsf.get("Strict"))
    brain_loc = as_number(ucsf.get("Loc"))
    lung_task = as_number(nsclc.get("Task"))
    lung_strict = as_number(nsclc.get("Strict"))
    lung_loc = as_number(nsclc.get("Loc (RAS)"))

    row = {
        "Model": record["paper_label"],
        "Category": record["category"],
        "Longitudinal Task": long_task,
        "Longitudinal Strict": long_strict,
        "Brain Task": brain_task,
        "Brain Strict": brain_strict,
        "Brain Loc.": brain_loc,
        "Lung Task": lung_task,
        "Lung Strict": lung_strict,
        "Lung Loc.": lung_loc,
        "Model Avg. Task": mean_present([long_task, brain_task, lung_task]),
        "Model Avg. Strict": mean_present([long_strict, brain_strict, lung_strict]),
        "Model Avg. Loc.": mean_present([brain_loc, lung_loc]),
    }
    detail = {
        "key": key,
        "model": record,
        "metrics": row,
        "source_files": {
            "lumiere": str(lumiere_source) if lumiere_source else None,
            "ucsf_pdgm": str(ucsf_source) if ucsf_source else None,
            "nsclc": str(nsclc_source) if nsclc_source else None,
        },
        "paper_table_data": {
            "lumiere": lumiere or None,
            "ucsf_pdgm": ucsf or None,
            "nsclc": nsclc or None,
        },
    }
    return row, detail


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", nargs="*", default=None, help="Model keys; defaults to every model directory present")
    parser.add_argument("--output-prefix", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    model_records = config["models"]
    model_keys = args.models or [
        key for key in config["sets"]["all"] if (results_root / key).is_dir()
    ]
    unknown = [key for key in model_keys if key not in model_records]
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")
    if not model_keys:
        raise SystemExit(f"No model result directories found under {results_root}")

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for key in model_keys:
        row, detail = build_row(key=key, record=model_records[key], results_root=results_root)
        rows.append(row)
        details.append(detail)

    prefix = (args.output_prefix or (results_root / "table1_results")).expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps({"schema_version": 1, "results_root": str(results_root), "models": details}, indent=2),
        encoding="utf-8",
    )
    print(csv_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
