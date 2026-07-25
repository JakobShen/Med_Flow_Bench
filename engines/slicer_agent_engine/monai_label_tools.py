from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .algorithm_runtime import MonaiLabelRuntime

DEFAULT_VISTA3D_LABEL_OPACITY = 0.3


@dataclass(frozen=True)
class MonaiToolResult:
    payload: Dict[str, Any]
    artifacts: Dict[str, str] = field(default_factory=dict)
    ok: bool = True
    error: Optional[str] = None


def _safe_file_token(value: Optional[str], *, default: str = "vista3d") -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or default)).strip("._")
    return token[:96] or default


def _coerce_points(points: Optional[Sequence[Sequence[int | float]]], *, name: str) -> List[List[float]]:
    if points is None:
        return []
    out: List[List[float]] = []
    for item in points:
        vals = [float(x) for x in item]
        if len(vals) != 3:
            raise ValueError(f"{name} entries must be length-3 points")
        out.append(vals)
    return out


def _coerce_int_list(values: Optional[Sequence[int | float]], *, name: str) -> List[int]:
    if values is None:
        return []
    return [int(x) for x in values]


def _catalog_entries_from_payload(payload: Any) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("labels", "label_names", "label_dict", "class_names", "classes"):
            if key in payload:
                entries.extend(_catalog_entries_from_payload(payload[key]))
        for key, value in payload.items():
            try:
                label_id = int(key)
            except Exception:
                continue
            if isinstance(value, dict):
                name = value.get("name") or value.get("label") or value.get("description")
                entry = {"id": label_id, **value}
                if name is not None:
                    entry["name"] = str(name)
                entries.append(entry)
            elif isinstance(value, str):
                entries.append({"id": label_id, "name": value})
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                raw_id = item.get("id", item.get("label", item.get("value", idx)))
                try:
                    label_id = int(raw_id)
                except Exception:
                    label_id = idx
                name = item.get("name") or item.get("label_name") or item.get("description")
                entry = {"id": label_id, **item}
                if name is not None:
                    entry["name"] = str(name)
                entries.append(entry)
            elif isinstance(item, str):
                entries.append({"id": idx, "name": item})
    dedup: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        try:
            dedup[int(entry["id"])] = entry
        except Exception:
            continue
    return [dedup[k] for k in sorted(dedup)]


class MonaiLabelToolService:
    """MONAI Label / VISTA3D workflow logic kept outside the core ToolContext."""

    def __init__(
        self,
        *,
        ctx: Any,
        runtime_factory: Callable[[], MonaiLabelRuntime],
    ) -> None:
        self.ctx = ctx
        self._runtime_factory = runtime_factory

    def _runtime(self) -> MonaiLabelRuntime:
        return self._runtime_factory()

    def status(self) -> MonaiToolResult:
        result = self._runtime().run_blocking(operation="info", inputs={})
        payload = {
            "ok": bool(result.ok),
            "server_url": result.outputs.get("server_url"),
            "info": result.outputs.get("info"),
            "error": result.error,
        }
        return MonaiToolResult(payload=payload, ok=bool(result.ok), error=result.error)

    def label_catalog(self, *, query: Optional[str] = None, limit: int = 50) -> MonaiToolResult:
        status = self._runtime().run_blocking(operation="info", inputs={})
        info = status.outputs.get("info") if status.ok else {}
        bundle_dir = None
        if isinstance(info, dict):
            models = info.get("models")
            if isinstance(models, dict):
                model_info = models.get("vista3d") or next(iter(models.values()), None)
                if isinstance(model_info, dict):
                    bundle_dir = model_info.get("bundle_dir")
        bundle_dir = bundle_dir or os.environ.get("VISTA3D_BUNDLE_DIR") or os.environ.get("MONAI_LABEL_BUNDLE_DIR")

        entries: List[Dict[str, Any]] = []
        searched: List[str] = []
        if bundle_dir:
            root = Path(str(bundle_dir)).expanduser()
            for path in list(root.glob("docs/*.json")) + list(root.glob("configs/*.json")):
                searched.append(str(path))
                try:
                    entries.extend(_catalog_entries_from_payload(json.loads(path.read_text(encoding="utf-8"))))
                except Exception:
                    continue
        if not entries:
            entries = [
                {"id": 23, "name": "lung tumor", "source": "fallback_vista3d_known_label"},
                {"id": 3, "name": "spleen", "source": "fallback_vista3d_known_label"},
            ]
        if query:
            q = str(query).strip().lower()
            entries = [entry for entry in entries if q in str(entry.get("name", "")).lower() or q in str(entry.get("id", "")).lower()]
        payload = {
            "ok": True,
            "query": query,
            "bundle_dir": str(bundle_dir) if bundle_dir else None,
            "searched": searched,
            "entries": entries[: max(1, int(limit))],
            "n_entries": len(entries),
            "notes": "For NSCLC CT lung tumor, use label_prompt=[23]. VISTA3D is a CT model; brain MRI use is experimental.",
        }
        return MonaiToolResult(payload=payload)

    def vista3d_segment(
        self,
        *,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        model: str = "vista3d",
        label_prompt: Optional[Sequence[int]] = None,
        points_ijk: Optional[Sequence[Sequence[int | float]]] = None,
        points_ras: Optional[Sequence[Sequence[int | float]]] = None,
        point_labels: Optional[Sequence[int]] = None,
        seed_view: Optional[str] = None,
        seed_bbox_1000: Optional[Sequence[int | float]] = None,
        seed_point_1000: Optional[Sequence[int | float]] = None,
        point_label: int = 1,
        segmentation_name: Optional[str] = None,
        size: int = 512,
        auto_show: bool = True,
        auto_center: bool = True,
        opacity: float = DEFAULT_VISTA3D_LABEL_OPACITY,
        screen_views: Optional[Sequence[str]] = None,
    ) -> MonaiToolResult:
        ctx = self.ctx
        labels = _coerce_int_list(label_prompt, name="label_prompt") if label_prompt is not None else []
        safe_model = str(model or "vista3d").replace("/", "-")
        out_dir = ctx.session.out_dir / "monai_label" / "vista3d"
        out_dir.mkdir(parents=True, exist_ok=True)
        export_path = out_dir / f"source_{ctx.session.session_id}.nii.gz"

        exported = ctx.client.exec_bridge(
            bridge_dir=ctx.bridge_dir,
            tool="export_volume_nifti",
            args={
                "volume_id": source_volume_id,
                "volume_name": source_volume_name,
                "output_path": str(export_path),
            },
            session_id=ctx.session.session_id,
            timeout_s=max(float(getattr(ctx.client, "timeout_s", 60.0)), 300.0),
        )
        image_path = str(exported.get("path") or export_path)
        image_id = Path(image_path).name.replace(".nii.gz", "").replace(".nii", "")

        prompt_points = _coerce_points(points_ijk, name="points_ijk")
        point_conversions: List[Dict[str, Any]] = []
        for ras in _coerce_points(points_ras, name="points_ras"):
            converted = ctx.client.exec_bridge(
                bridge_dir=ctx.bridge_dir,
                tool="ras_to_volume_ijk",
                args={
                    "volume_id": exported.get("volume_id"),
                    "volume_name": exported.get("volume_name"),
                    "ras": ras,
                },
                session_id=ctx.session.session_id,
            )
            prompt_points.append([float(x) for x in converted.get("ijk_float", converted.get("ijk", []))])
            point_conversions.append({"source": "points_ras", "ras": ras, "converted": converted})

        if seed_view and (seed_bbox_1000 is not None or seed_point_1000 is not None):
            seed_ras_result = ctx.image_coords_to_ras(
                view=str(seed_view),
                bbox_1000=seed_bbox_1000,
                point_1000=seed_point_1000,
                size=int(size),
            )
            seed_ras = seed_ras_result.get("center_ras") or seed_ras_result.get("ras")
            if not seed_ras:
                raise RuntimeError(f"image_coords_to_ras did not return a seed RAS point: {seed_ras_result}")
            converted = ctx.client.exec_bridge(
                bridge_dir=ctx.bridge_dir,
                tool="ras_to_volume_ijk",
                args={
                    "volume_id": exported.get("volume_id"),
                    "volume_name": exported.get("volume_name"),
                    "ras": seed_ras,
                },
                session_id=ctx.session.session_id,
            )
            prompt_points.append([float(x) for x in converted.get("ijk_float", converted.get("ijk", []))])
            point_conversions.append({"source": "seed_screen", "seed_ras": seed_ras_result, "converted": converted})

        prompt_point_labels = _coerce_int_list(point_labels, name="point_labels")
        if prompt_points and not prompt_point_labels:
            prompt_point_labels = [int(point_label)] * len(prompt_points)
        if len(prompt_point_labels) != len(prompt_points):
            raise ValueError("point_labels must have the same length as points")
        if prompt_points and labels and len(labels) != 1:
            raise ValueError("VISTA3D point prompts can be combined with at most one label_prompt")

        params: Dict[str, Any] = {}
        if labels:
            params["label_prompt"] = labels
        if prompt_points:
            params["points"] = prompt_points
            params["point_labels"] = prompt_point_labels

        runtime_result = self._runtime().run_blocking(
            operation="vista3d_segment",
            inputs={
                "model": model or "vista3d",
                "image_id": image_id,
                "image_path": image_path,
                "params": params,
                "output_dir": out_dir / safe_model,
            },
        )
        if not runtime_result.ok:
            payload = {
                "ok": False,
                "error": runtime_result.error,
                "exported_volume": exported,
                "model": model,
                "prompt_params": params,
            }
            return MonaiToolResult(payload=payload, ok=False, error=runtime_result.error)

        label_path = runtime_result.outputs.get("label_path")
        if not label_path:
            raise RuntimeError(f"MONAI Label did not return a label_path: {runtime_result.outputs}")

        imported = ctx.client.exec_bridge(
            bridge_dir=ctx.bridge_dir,
            tool="load_labelmap_as_segmentation",
            args={
                "labelmap_path": str(label_path),
                "reference_volume_id": exported.get("volume_id"),
                "reference_volume_name": exported.get("volume_name"),
                "segmentation_name": segmentation_name or (f"VISTA3D {','.join(str(x) for x in labels)}" if labels else "VISTA3D"),
                "remove_labelmap": False,
                "show_labelmap_in_viewer": bool(auto_show),
                "visible_label_values": labels or None,
                "label_opacity": float(opacity),
            },
            session_id=ctx.session.session_id,
            timeout_s=max(float(getattr(ctx.client, "timeout_s", 60.0)), 300.0),
        )
        if isinstance(imported, dict):
            imported = dict(imported)

        seg_id = imported.get("segmentation_id") if isinstance(imported, dict) else None
        seg_name = imported.get("segmentation_name") if isinstance(imported, dict) else None
        segment_ids = list(imported.get("segment_ids") or []) if isinstance(imported, dict) else []
        segment_names = list(imported.get("segment_names") or []) if isinstance(imported, dict) else []
        target_segment_id = imported.get("segment_id") if isinstance(imported, dict) else None
        target_segment_name = imported.get("segment_name") if isinstance(imported, dict) else None
        if labels and segment_ids:
            wanted = {str(v) for v in labels}
            for sid, sname in zip(segment_ids, segment_names):
                if str(sname) in wanted:
                    target_segment_id = sid
                    target_segment_name = sname
                    break
        if isinstance(imported, dict):
            imported["segment_id"] = target_segment_id
            imported["segment_name"] = target_segment_name
            ctx._apply_segmentation_state_from_result(imported)

        show_result = None
        center_result = None
        if bool(auto_show) and labels and (seg_id or seg_name):
            for sid, sname in zip(segment_ids, segment_names):
                if sid != target_segment_id:
                    ctx.hide_segmentation(
                        segmentation_id=seg_id,
                        segmentation_name=seg_name,
                        segment_id=sid,
                        segment_name=sname,
                    )
        if bool(auto_show) and (seg_id or seg_name):
            show_result = ctx.show_segmentation(
                segmentation_id=seg_id,
                segmentation_name=seg_name,
                segment_id=target_segment_id,
                segment_name=target_segment_name,
                visible=True,
                fill=True,
                outline=True,
                opacity=float(opacity),
                color=[0.0, 1.0, 0.0],
            )
        if bool(auto_center) and (seg_id or seg_name):
            center_result = ctx.center_on_segment(
                segmentation_id=seg_id,
                segmentation_name=seg_name,
                segment_id=target_segment_id,
                segment_name=target_segment_name,
            )

        viewer_observation = None
        png_paths: Dict[str, str] = {}
        if bool(auto_show):
            view_token = _safe_file_token(
                segmentation_name
                or (f"vista3d_label_{'_'.join(str(x) for x in labels)}" if labels else "vista3d_segment_everything")
            )
            viewer_observation = ctx.observe_standard_views(size=int(size), out_prefix=f"{view_token}_viewer")
            if isinstance(viewer_observation, dict) and isinstance(viewer_observation.get("png_paths"), dict):
                png_paths = dict(viewer_observation["png_paths"])

        payload = {
            "ok": True,
            "model": model,
            "label_prompt": labels if labels else None,
            "prompt_params": params,
            "point_conversions": point_conversions,
            "exported_volume": exported,
            "label_path": str(label_path),
            "segmentation": imported,
            "segmentation_id": seg_id,
            "segmentation_name": seg_name,
            "segment_ids": imported.get("segment_ids") if isinstance(imported, dict) else None,
            "segment_names": imported.get("segment_names") if isinstance(imported, dict) else None,
            "segment_id": target_segment_id,
            "segment_name": target_segment_name,
            "show": show_result,
            "center": center_result,
            "viewer_observation": viewer_observation,
            "png_paths": png_paths,
            "png_path": png_paths.get("axial"),
        }
        vista3d_artifacts = dict(ctx.session.state.get("vista3d_segmentation_artifacts") or {})
        if seg_id:
            vista3d_artifacts[str(seg_id)] = {
                "label_values": labels or None,
                "segmentation_name": seg_name,
                "labelmap_volume_id": imported.get("labelmap_volume_id") if isinstance(imported, dict) else None,
                "labelmap_volume_name": imported.get("labelmap_volume_name") if isinstance(imported, dict) else None,
                "source_volume_id": exported.get("volume_id"),
                "source_volume_name": exported.get("volume_name"),
            }
        ctx.session.update_state(
            last_vista3d_segmentation_id=seg_id,
            last_vista3d_segmentation_name=seg_name,
            vista3d_segmentation_artifacts=vista3d_artifacts,
        )
        artifacts = {"nifti": image_path, "labelmap": str(label_path)}
        if png_paths.get("axial"):
            artifacts["png"] = png_paths["axial"]
        return MonaiToolResult(payload=payload, artifacts=artifacts)

    def capture_segmentation_overlay(
        self,
        *,
        view: str = "axial",
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
        out_name: Optional[str] = None,
    ) -> MonaiToolResult:
        ctx = self.ctx
        ctx.show_segmentation(
            segmentation_id=segmentation_id,
            segmentation_name=segmentation_name,
            segment_id=segment_id,
            segment_name=segment_name,
            visible=True,
        )
        center = ctx.center_on_segment(
            segmentation_id=segmentation_id,
            segmentation_name=segmentation_name,
            segment_id=segment_id,
            segment_name=segment_name,
        )
        observed = ctx.observe_view(
            view=view,
            out_name=out_name or f"segmentation_overlay_{str(view or 'axial').lower()}.png",
        )
        payload = {"ok": True, "source": "slicer_viewer", "center": center, **observed}
        artifacts = {"png": str(payload.get("png_path"))} if payload.get("png_path") else {}
        return MonaiToolResult(payload=payload, artifacts=artifacts)
