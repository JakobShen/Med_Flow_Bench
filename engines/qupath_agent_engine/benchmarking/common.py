from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any, Iterable

from qupath_agent_engine.agent.tooling import infer_tool_category


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def normalize_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric > 1.0 and numeric <= 100.0:
        numeric /= 100.0
    if not (0.0 <= numeric <= 1.0):
        return None
    return numeric


def collect_step_stats(steps: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = [dict(item) for item in steps]
    n_direct_tools = sum(1 for step in rows if infer_tool_category(step.get("tool")) == "direct")
    n_gui_tools = sum(1 for step in rows if infer_tool_category(step.get("tool")) == "gui")
    n_evidence_tools = sum(1 for step in rows if infer_tool_category(step.get("tool")) == "evidence")
    return {
        "n_steps": len(rows),
        "n_direct_tools": n_direct_tools,
        "n_gui_tools": n_gui_tools,
        "n_evidence_tools": n_evidence_tools,
        "n_tool_calls": n_direct_tools + n_gui_tools + n_evidence_tools,
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def safe_mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    return mean(numbers)


def normalize_output_localizations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        parsed = _normalize_single_output_localization(item)
        if parsed is None:
            continue
        parsed.setdefault("localization_id", f"localization_{index:04d}")
        normalized.append(parsed)
    return normalized


def build_output_localization_records(
    localizations: Iterable[dict[str, Any]],
    *,
    prefix: str = "output_localization",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, localization in enumerate(localizations, start=1):
        record = dict(localization)
        x = record.get("x")
        y = record.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        records.append(
            {
                "evidence_id": str(record.get("localization_id") or f"{prefix}_{index:04d}"),
                "center_pixel_x": float(x),
                "center_pixel_y": float(y),
                "supported_label": record.get("supported_label"),
                "description": record.get("description"),
                "localization": record,
            }
        )
    return records


def _normalize_single_output_localization(value: Any) -> dict[str, Any] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        x = _coerce_float(value[0])
        y = _coerce_float(value[1])
        if x is None or y is None:
            return None
        return {"x": x, "y": y}

    if not isinstance(value, dict):
        return None

    x = _first_float(
        value,
        "x",
        "image_x",
        "wsi_x",
        "center_pixel_x",
        "pixel_x",
    )
    y = _first_float(
        value,
        "y",
        "image_y",
        "wsi_y",
        "center_pixel_y",
        "pixel_y",
    )
    if x is None or y is None:
        return None

    normalized: dict[str, Any] = {
        "x": x,
        "y": y,
    }
    localization_id = value.get("localization_id") or value.get("id") or value.get("evidence_id")
    if localization_id is not None:
        normalized["localization_id"] = str(localization_id).strip()
    supported_label = value.get("supported_label")
    if supported_label is not None and str(supported_label).strip():
        normalized["supported_label"] = str(supported_label).strip()
    description = value.get("description")
    if description is not None and str(description).strip():
        normalized["description"] = str(description).strip()
    return normalized


def _first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        candidate = _coerce_float(mapping.get(key))
        if candidate is not None:
            return candidate
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not (numeric == numeric):
        return None
    if numeric in (float("inf"), float("-inf")):
        return None
    return numeric
