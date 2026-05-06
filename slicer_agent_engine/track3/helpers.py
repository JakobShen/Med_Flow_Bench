from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, TypeVar


T = TypeVar("T")


_METHOD_ALIASES = {
    "figure": "compressed_figures",
    "figures": "compressed_figures",
    "compressed": "compressed_figures",
    "compressed_figures": "compressed_figures",
    "montage": "compressed_figures",
    "table3": "table3_api",
    "table3_api": "table3_api",
    "sampled": "table3_api",
    "sampled_slices": "table3_api",
    "gpt": "table3_api",
    "m3d": "m3d",
    "radfm": "radfm",
}


_METHOD_DISPLAY_NAMES = {
    "compressed_figures": "compressed figures baseline",
    "table3_api": "Table-3-style sampled-slices baseline",
    "m3d": "M3D",
    "radfm": "RadFM",
}


_SAFE_STEM_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd", ".npz", ".npy", ".json", ".png", ".jpg", ".jpeg")


class Track3ConfigError(ValueError):
    pass


def normalize_method_name(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_")
    if key not in _METHOD_ALIASES:
        allowed = ", ".join(sorted(set(_METHOD_ALIASES.values())))
        raise Track3ConfigError(f"Unsupported Track3 method: {name!r}. Supported: {allowed}")
    return _METHOD_ALIASES[key]


def method_display_name(name: str) -> str:
    canonical = normalize_method_name(name)
    return _METHOD_DISPLAY_NAMES.get(canonical, canonical)


def parse_csv_list(raw: str | Sequence[str] | None, *, default: Optional[Sequence[str]] = None) -> List[str]:
    if raw is None:
        return [str(x).strip() for x in (default or []) if str(x).strip()]
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",")]
    else:
        items = [str(part).strip() for part in raw]
    out: List[str] = []
    for item in items:
        if item:
            out.append(item)
    return out


def safe_stem(path: str | Path) -> str:
    name = Path(path).name
    lower = name.lower()
    for suffix in _SAFE_STEM_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def safe_token(value: str, *, default: str = "unknown") -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "").strip())
    token = token.strip("-._")
    return token or default


def ensure_dir(path: str | Path) -> Path:
    path_p = Path(path).expanduser().resolve()
    path_p.mkdir(parents=True, exist_ok=True)
    return path_p


def uniform_indices(length: int, count: int) -> List[int]:
    length_i = max(0, int(length))
    count_i = max(0, int(count))
    if length_i <= 0 or count_i <= 0:
        return []
    if count_i >= length_i:
        return list(range(length_i))
    if count_i == 1:
        return [int(round((length_i - 1) / 2.0))]
    out = []
    for idx in range(count_i):
        pos = idx * (length_i - 1) / float(count_i - 1)
        out.append(int(round(pos)))
    deduped: List[int] = []
    seen = set()
    for idx in out:
        idx = min(max(int(idx), 0), length_i - 1)
        if idx in seen:
            continue
        deduped.append(idx)
        seen.add(idx)
    if len(deduped) < count_i:
        for idx in range(length_i):
            if idx in seen:
                continue
            deduped.append(idx)
            seen.add(idx)
            if len(deduped) >= count_i:
                break
    return deduped[:count_i]


def uniform_downsample(items: Sequence[T], max_items: int) -> List[T]:
    max_items_i = max(0, int(max_items))
    seq = list(items)
    if max_items_i <= 0:
        return []
    if len(seq) <= max_items_i:
        return seq
    return [seq[idx] for idx in uniform_indices(len(seq), max_items_i)]


def chunked(items: Sequence[T], chunk_size: int) -> List[List[T]]:
    chunk_i = max(1, int(chunk_size))
    seq = list(items)
    return [seq[idx : idx + chunk_i] for idx in range(0, len(seq), chunk_i)]


def majority_vote(
    predictions: Sequence[Tuple[str, Optional[str]]],
    *,
    valid_choices: Sequence[str] = ("A", "B", "C"),
) -> Tuple[Optional[str], dict[str, int]]:
    allowed = {str(choice).strip().upper() for choice in valid_choices if str(choice).strip()}
    ordered = [(str(label), (choice or "").strip().upper() or None) for label, choice in predictions]
    counts = Counter(choice for _, choice in ordered if choice in allowed)
    if not counts:
        return None, {}
    best = max(counts.values())
    winners = {choice for choice, n in counts.items() if n == best}
    for _, choice in ordered:
        if choice in winners:
            return choice, {k: int(v) for k, v in sorted(counts.items())}
    return None, {k: int(v) for k, v in sorted(counts.items())}


def pick_first_existing(paths: Iterable[str | Path | None]) -> Optional[Path]:
    for value in paths:
        if value is None:
            continue
        path = Path(value).expanduser().resolve()
        if path.exists():
            return path
    return None


def env_or_default(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.environ.get(str(name))
        if value:
            return value
    return default


__all__ = [
    "Track3ConfigError",
    "chunked",
    "ensure_dir",
    "env_or_default",
    "majority_vote",
    "method_display_name",
    "normalize_method_name",
    "parse_csv_list",
    "pick_first_existing",
    "safe_stem",
    "safe_token",
    "uniform_downsample",
    "uniform_indices",
]
