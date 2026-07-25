from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .helpers import chunked, ensure_dir, safe_token, uniform_downsample
from .volumes import LoadedVolume


_PLANE_SHORT = {
    "axial": "Ax",
    "sagittal": "Sag",
    "coronal": "Cor",
}


@dataclass(frozen=True)
class SliceTile:
    volume_name: str
    volume_path: str
    sequence_label: str
    plane: str
    slice_index: int
    axis_label: str
    axis_mm: float


@dataclass(frozen=True)
class TrackCArtifact:
    path: Path
    description: str
    kind: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactBundle:
    artifacts: List[TrackCArtifact]
    manifest_path: Path
    manifest: List[Dict[str, Any]]


_MODALITY_HINTS = (
    ("t1c", "T1c"),
    ("t1ce", "T1c"),
    ("ct1", "CT1"),
    ("post", "T1c"),
    ("contrast", "T1c"),
    ("flair", "FLAIR"),
    ("t2", "T2"),
    ("t1", "T1"),
    ("dwi", "DWI"),
    ("adc", "ADC"),
    ("swi", "SWI"),
)


def _sequence_label(volume: LoadedVolume) -> str:
    name_l = volume.stem.lower()
    side_prefix = ""
    side_match = re.match(r"^(old|current)(?:[_\-\s]|$)", volume.stem, flags=re.IGNORECASE)
    if side_match:
        side_prefix = side_match.group(1).upper()
    for token, label in _MODALITY_HINTS:
        if token in name_l:
            return f"{side_prefix} {label}".strip()
    base = volume.stem.replace("_", " ").replace("-", " ").strip()
    if len(base) <= 18:
        return base
    return base[:15].rstrip() + "…"


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Track C artifact generation requires Pillow. Install it in the Track C environment, e.g. `pip install Pillow`."
        ) from e
    return Image, ImageDraw, ImageFont, ImageOps


def _load_font(size: int):
    Image, ImageDraw, ImageFont, ImageOps = _load_pillow()
    del Image, ImageDraw, ImageOps
    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _tile_from_spec(
    volume: LoadedVolume,
    spec: SliceTile,
    *,
    tile_size: Tuple[int, int] = (220, 220),
    caption_height: int = 48,
):
    Image, ImageDraw, ImageFont, ImageOps = _load_pillow()
    tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    img_h = max(16, tile_h - caption_height)
    font_main = _load_font(22)
    font_caption = _load_font(18)

    arr = volume.slice_2d_uint8(spec.plane, spec.slice_index)
    base = Image.fromarray(arr, mode="L").convert("RGB")
    canvas = Image.new("RGB", (tile_w, tile_h), color=(0, 0, 0))

    fitted = ImageOps.contain(base, (tile_w - 8, img_h - 8))
    offset_x = (tile_w - fitted.size[0]) // 2
    offset_y = (img_h - fitted.size[1]) // 2
    canvas.paste(fitted, (offset_x, offset_y))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, tile_w - 1, tile_h - 1], outline=(160, 160, 160), width=1)
    draw.rectangle([0, img_h, tile_w - 1, tile_h - 1], fill=(18, 18, 18))

    line1 = f"{spec.sequence_label} { _PLANE_SHORT.get(spec.plane, spec.plane[:3].title()) }"
    line2 = f"{spec.axis_label}={spec.axis_mm:.1f} mm"
    draw.text((8, img_h + 4), line1, font=font_main, fill=(255, 255, 255))
    draw.text((8, img_h + 24), line2, font=font_caption, fill=(220, 220, 220))
    return canvas


def _build_contact_sheet(
    *,
    figure_path: Path,
    title: str,
    subtitle: str,
    tiles: Sequence[Tuple[LoadedVolume, SliceTile]],
    cols: int = 6,
    tile_size: Tuple[int, int] = (220, 220),
    header_height: int = 74,
) -> Path:
    Image, ImageDraw, ImageFont, ImageOps = _load_pillow()
    del ImageOps
    cols_i = max(1, int(cols))
    rows = int(np.ceil(len(tiles) / float(cols_i))) if tiles else 1
    tile_w, tile_h = int(tile_size[0]), int(tile_size[1])
    width = cols_i * tile_w + (cols_i + 1) * 8
    height = rows * tile_h + (rows + 1) * 8 + int(header_height)
    canvas = Image.new("RGB", (width, height), color=(10, 10, 10))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(30)
    subtitle_font = _load_font(18)
    draw.text((10, 10), title, font=title_font, fill=(255, 255, 255))
    draw.text((10, 44), subtitle, font=subtitle_font, fill=(200, 200, 200))

    for idx, (volume, spec) in enumerate(tiles):
        row = idx // cols_i
        col = idx % cols_i
        x = 8 + col * (tile_w + 8)
        y = int(header_height) + 8 + row * (tile_h + 8)
        canvas.paste(_tile_from_spec(volume, spec, tile_size=tile_size), (x, y))

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure_path)
    return figure_path


def _iter_all_slice_specs(volumes: Sequence[LoadedVolume], planes: Sequence[str]) -> List[Tuple[LoadedVolume, SliceTile]]:
    out: List[Tuple[LoadedVolume, SliceTile]] = []
    for volume in volumes:
        seq_label = _sequence_label(volume)
        for plane in planes:
            depth = volume.depth_for_plane(plane)
            for idx in range(depth):
                axis_label, axis_mm = volume.slice_axis_label_and_mm(plane, idx)
                out.append(
                    (
                        volume,
                        SliceTile(
                            volume_name=volume.name,
                            volume_path=str(volume.path),
                            sequence_label=seq_label,
                            plane=str(plane),
                            slice_index=int(idx),
                            axis_label=axis_label,
                            axis_mm=float(axis_mm),
                        ),
                    )
                )
    return out


def build_compressed_figures(
    volumes: Sequence[LoadedVolume],
    *,
    out_dir: str | Path,
    case_id: str,
    planes: Sequence[str],
    slices_per_figure: int = 42,
    max_figures: int = 9,
    cols: int = 6,
) -> ArtifactBundle:
    out_dir_p = ensure_dir(out_dir)
    max_total_tiles = max(1, int(slices_per_figure)) * max(1, int(max_figures))
    all_specs = _iter_all_slice_specs(volumes, planes)
    kept_specs = uniform_downsample(all_specs, max_total_tiles)
    groups = chunked(kept_specs, max(1, int(slices_per_figure)))

    artifacts: List[TrackCArtifact] = []
    manifest: List[Dict[str, Any]] = []
    for fig_idx, group in enumerate(groups, start=1):
        figure_path = out_dir_p / f"figure_{fig_idx:02d}.png"
        title = f"{case_id} • Figure {fig_idx}/{len(groups)}"
        subtitle = "Tile caption format: <sequence> <plane> | canonical axis coordinate in mm"
        _build_contact_sheet(
            figure_path=figure_path,
            title=title,
            subtitle=subtitle,
            tiles=group,
            cols=cols,
        )
        figure_manifest = {
            "figure_index": int(fig_idx),
            "path": str(figure_path),
            "kind": "compressed_figure",
            "n_tiles": len(group),
            "tiles": [
                {
                    "volume_name": spec.volume_name,
                    "volume_path": spec.volume_path,
                    "sequence_label": spec.sequence_label,
                    "plane": spec.plane,
                    "slice_index": int(spec.slice_index),
                    "axis_label": spec.axis_label,
                    "axis_mm": float(spec.axis_mm),
                }
                for _, spec in group
            ],
        }
        manifest.append(figure_manifest)
        artifacts.append(
            TrackCArtifact(
                path=figure_path,
                kind="compressed_figure",
                description=(
                    f"Compressed figure {fig_idx}/{len(groups)} for {case_id}. "
                    "Each tile caption shows sequence, plane, and canonical axis coordinate in mm."
                ),
                metadata=figure_manifest,
            )
        )

    manifest_path = out_dir_p / "compressed_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ArtifactBundle(artifacts=artifacts, manifest_path=manifest_path, manifest=manifest)


def build_table3_artifacts(
    volumes: Sequence[LoadedVolume],
    *,
    out_dir: str | Path,
    case_id: str,
    planes: Sequence[str],
    sample_slices: int = 10,
    delivery: str = "montage",
    cols: int = 5,
) -> ArtifactBundle:
    out_dir_p = ensure_dir(out_dir)
    delivery_l = str(delivery).strip().lower()
    if delivery_l not in {"montage", "images"}:
        raise ValueError(f"Unsupported table3 delivery: {delivery!r}")

    artifacts: List[TrackCArtifact] = []
    manifest: List[Dict[str, Any]] = []
    entry_idx = 0
    for volume in volumes:
        seq_label = _sequence_label(volume)
        for plane in planes:
            indices = volume.sample_indices(plane, sample_slices)
            group: List[Tuple[LoadedVolume, SliceTile]] = []
            for idx in indices:
                axis_label, axis_mm = volume.slice_axis_label_and_mm(plane, idx)
                group.append(
                    (
                        volume,
                        SliceTile(
                            volume_name=volume.name,
                            volume_path=str(volume.path),
                            sequence_label=seq_label,
                            plane=str(plane),
                            slice_index=int(idx),
                            axis_label=axis_label,
                            axis_mm=float(axis_mm),
                        ),
                    )
                )

            if delivery_l == "montage":
                entry_idx += 1
                plane_token = safe_token(plane)
                seq_token = safe_token(seq_label)
                path = out_dir_p / f"sampled_{entry_idx:02d}_{seq_token}_{plane_token}.png"
                title = f"{case_id} • {seq_label} • {plane.title()}"
                subtitle = f"Table-3-style uniform sampling: {len(group)} slices"
                _build_contact_sheet(
                    figure_path=path,
                    title=title,
                    subtitle=subtitle,
                    tiles=group,
                    cols=cols,
                )
                item_manifest = {
                    "path": str(path),
                    "kind": "table3_montage",
                    "volume_name": volume.name,
                    "plane": plane,
                    "n_tiles": len(group),
                    "tiles": [
                        {
                            "slice_index": int(spec.slice_index),
                            "axis_label": spec.axis_label,
                            "axis_mm": float(spec.axis_mm),
                        }
                        for _, spec in group
                    ],
                }
                manifest.append(item_manifest)
                artifacts.append(
                    TrackCArtifact(
                        path=path,
                        kind="table3_montage",
                        description=(
                            f"Uniformly sampled {len(group)} {plane} slices from sequence {seq_label}. "
                            "Each tile caption shows the canonical axis coordinate in mm."
                        ),
                        metadata=item_manifest,
                    )
                )
            else:
                seq_token = safe_token(seq_label)
                plane_token = safe_token(plane)
                for local_idx, (volume_obj, spec) in enumerate(group, start=1):
                    entry_idx += 1
                    path = out_dir_p / f"sampled_{entry_idx:03d}_{seq_token}_{plane_token}_{local_idx:02d}.png"
                    tile = _tile_from_spec(volume_obj, spec, tile_size=(360, 360), caption_height=56)
                    tile.save(path)
                    item_manifest = {
                        "path": str(path),
                        "kind": "table3_image",
                        "volume_name": volume.name,
                        "plane": plane,
                        "slice_index": int(spec.slice_index),
                        "axis_label": spec.axis_label,
                        "axis_mm": float(spec.axis_mm),
                    }
                    manifest.append(item_manifest)
                    artifacts.append(
                        TrackCArtifact(
                            path=path,
                            kind="table3_image",
                            description=(
                                f"Single sampled slice from sequence {seq_label}, plane {plane}, "
                                f"at canonical coordinate {spec.axis_label}={spec.axis_mm:.1f} mm."
                            ),
                            metadata=item_manifest,
                        )
                    )

    manifest_path = out_dir_p / "table3_artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return ArtifactBundle(artifacts=artifacts, manifest_path=manifest_path, manifest=manifest)


__all__ = [
    "ArtifactBundle",
    "SliceTile",
    "TrackCArtifact",
    "build_compressed_figures",
    "build_table3_artifacts",
]
