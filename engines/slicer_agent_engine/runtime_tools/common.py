
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union, TYPE_CHECKING

from ..click_contract import annotate_png_with_bbox, annotate_png_with_corner_text, annotate_png_with_point, crop_png_to_focus
from ..context_briefs import format_scene_inventory_for_prompt

if TYPE_CHECKING:
    from ..algorithm_runtime import AlgorithmRuntime
    from ..gemini_video import GeminiVideoAnalyzer

logger = logging.getLogger(__name__)


def _as_path(p: Union[str, Path]) -> Path:
    # Keep staged symlink basenames intact for downstream loaders that derive scene node names from filenames.
    return Path(os.path.abspath(os.path.expanduser(str(p))))


_VIEW_ALIASES = {
    "axial": "axial",
    "red": "axial",
    "r": "axial",
    "sagittal": "sagittal",
    "yellow": "sagittal",
    "y": "sagittal",
    "coronal": "coronal",
    "green": "coronal",
    "g": "coronal",
}
_STANDARD_VIEWS = ("axial", "sagittal", "coronal")
_DEFAULT_CANONICAL_AXIS_LABEL_BY_VIEW = {"axial": "S", "sagittal": "R", "coronal": "A"}

_VIEWER_MUTATION_TOOLS = {
    "select_volume",
    "set_window_level",
    "apply_window_preset",
    "set_interpolation",
    "set_slice_orientation",
    "set_slice_offset",
    "set_slice_scroll_to",
    "fit_slice",
    "zoom_slice_relative",
    "set_field_of_view",
    "set_xyz_origin",
    "jump_to_ras",
    "recover_standard_views",
    "set_linked_slices",
    "set_layout",
    "set_thick_slab",
    "set_fusion",
    "clear_fusion",
    "compute_subtraction",
    "center_on_segment",
}

_SCENE_MUTATION_TOOLS = {
    "clear_scene",
    "load_dicom",
    "load_dicom_series",
    "load_nifti",
    "open_case",
    "compute_subtraction",
}


def _canonical_view(view: Optional[str], *, default: str = "axial") -> str:
    v = (view or default).strip().lower()
    if v in _VIEW_ALIASES:
        return _VIEW_ALIASES[v]
    raise ValueError(f"Unsupported view: {view!r} (expected axial/sagittal/coronal)")


def _normalize_public_view_payload(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload

    normalized = dict(payload)
    view = normalized.get("view")
    if isinstance(view, str):
        try:
            normalized["view"] = _canonical_view(view)
        except ValueError:
            pass

    views = normalized.get("views")
    if isinstance(views, dict):
        normalized_views: Dict[str, Any] = {}
        for key, value in views.items():
            if isinstance(key, str):
                try:
                    normalized_views[_canonical_view(key)] = value
                    continue
                except ValueError:
                    pass
            normalized_views[key] = value
        normalized["views"] = normalized_views

    return normalized


class ToolContextBaseMixin:
    def _log(
        self,
        *,
        tool: str,
        args: Dict[str, Any],
        result: Any,
        ok: bool = True,
        error: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        self._maybe_refresh_state_before_log(tool)
        self.session.log_event(tool=tool, args=args, result=result, ok=ok, error=error, artifacts=artifacts)

    def _artifact_path(self, name: str) -> Path:
        p = self.session.out_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _estimate_load_timeout_s(self, *, n_series: int = 1) -> float:
        n = max(int(n_series), 1)
        return float(max(self.client.timeout_s, min(900.0, 90.0 + 20.0 * n)))

    def _annotate_bbox_preview(self, *, png_path: str, bbox_1000: Sequence[int | float], out_name: str, label: Optional[str] = None) -> str:
        out_path = self._artifact_path(out_name)
        annotate_png_with_bbox(png_path, bbox_1000=bbox_1000, out_path=out_path, label=label)
        return str(out_path)

    def _annotate_point_preview(self, *, png_path: str, point_1000: Sequence[int | float], out_name: str, label: Optional[str] = None) -> str:
        out_path = self._artifact_path(out_name)
        annotate_png_with_point(png_path, point_1000=point_1000, out_path=out_path, label=label)
        return str(out_path)

    def _crop_focus_png_in_place(self, *, png_path: str) -> tuple[str, tuple[int, int, int, int], tuple[int, int]]:
        cropped_path, crop_box, full_size = crop_png_to_focus(png_path)
        return str(cropped_path), crop_box, full_size

    def _format_slice_mm_overlay_line(self, *, view: str, offset_mm: Optional[float], axis_label: Optional[str] = None) -> Optional[str]:
        if offset_mm is None:
            return None
        try:
            value = float(offset_mm)
        except Exception:
            return None
        canonical_view = _canonical_view(view)
        label = str(axis_label or _DEFAULT_CANONICAL_AXIS_LABEL_BY_VIEW.get(canonical_view) or canonical_view).strip().upper()
        return f"{canonical_view} {label}: {value:.4f} mm"

    def _stamp_png_corner_text(self, *, png_path: str, lines: Sequence[str], corner: str = "top_left") -> str:
        try:
            annotate_png_with_corner_text(png_path, lines=lines, out_path=png_path, corner=corner)
        except Exception:
            logger.debug("Failed stamping corner text on PNG: %s", png_path, exc_info=True)
        return str(png_path)

    def _quiet_get_viewer_state(self) -> Optional[Dict[str, Any]]:
        cached = self.session.state.get("viewer_state") if hasattr(self, "session") else None
        if isinstance(cached, dict) and cached:
            return _normalize_public_view_payload(cached)
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_viewer_state",
                args={},
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            if isinstance(result, dict) and result:
                self.session.update_state(viewer_state=result)
                return result
        except Exception:
            logger.debug("quiet get_viewer_state failed", exc_info=True)
        return None

    def _quiet_get_slice_offset_range(self, *, view: str) -> Optional[Dict[str, Any]]:
        canonical_view = _canonical_view(view)
        cached = self.session.state.get("viewer_state") if hasattr(self, "session") else None
        if isinstance(cached, dict):
            views = cached.get("views")
            if isinstance(views, dict):
                payload = views.get(canonical_view)
                if isinstance(payload, dict) and payload.get("canonical_min_offset") is not None and payload.get("canonical_max_offset") is not None:
                    return {
                        "view": canonical_view,
                        "min_offset": payload.get("canonical_min_offset"),
                        "max_offset": payload.get("canonical_max_offset"),
                        "axis_label": payload.get("canonical_axis_label") or _DEFAULT_CANONICAL_AXIS_LABEL_BY_VIEW.get(canonical_view),
                        "canonical_sign": payload.get("canonical_sign"),
                    }
                if isinstance(payload, dict) and payload.get("min_offset") is not None and payload.get("max_offset") is not None:
                    return {
                        "view": canonical_view,
                        "min_offset": payload.get("min_offset"),
                        "max_offset": payload.get("max_offset"),
                        "axis_label": payload.get("canonical_axis_label") or _DEFAULT_CANONICAL_AXIS_LABEL_BY_VIEW.get(canonical_view),
                        "canonical_sign": payload.get("canonical_sign"),
                    }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_slice_offset_range",
                args={"view": canonical_view},
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            if isinstance(result, dict) and result:
                if result.get("canonical_min_offset") is not None and result.get("canonical_max_offset") is not None:
                    return {
                        **result,
                        "view": canonical_view,
                        "min_offset": result.get("canonical_min_offset"),
                        "max_offset": result.get("canonical_max_offset"),
                        "axis_label": result.get("canonical_axis_label") or _DEFAULT_CANONICAL_AXIS_LABEL_BY_VIEW.get(canonical_view),
                        "canonical_sign": result.get("canonical_sign"),
                    }
                result.setdefault("axis_label", result.get("canonical_axis_label") or _DEFAULT_CANONICAL_AXIS_LABEL_BY_VIEW.get(canonical_view))
                result.setdefault("canonical_sign", result.get("canonical_sign"))
                return result
        except Exception:
            logger.debug("quiet get_slice_offset_range failed for view=%s", canonical_view, exc_info=True)
        return None

    def _stamp_slice_mm_on_png(
        self,
        *,
        png_path: str,
        view: str,
        offset: Optional[float] = None,
        scroll_to: Optional[float] = None,
        allow_viewer_state_fallback: bool = True,
        canonical_offset: Optional[float] = None,
        axis_label: Optional[str] = None,
    ) -> str:
        canonical_view = _canonical_view(view)
        effective_offset: Optional[float] = None
        effective_axis_label: Optional[str] = str(axis_label).strip().upper() if axis_label else None
        if canonical_offset is not None:
            try:
                effective_offset = float(canonical_offset)
            except Exception:
                effective_offset = None
        elif offset is not None and scroll_to is None:
            offset_range = self._quiet_get_slice_offset_range(view=canonical_view)
            if isinstance(offset_range, dict):
                if effective_axis_label is None:
                    axis_label_raw = offset_range.get("axis_label")
                    if axis_label_raw is not None:
                        effective_axis_label = str(axis_label_raw).strip().upper()
                sign_raw = offset_range.get("canonical_sign")
                try:
                    effective_offset = float(sign_raw if sign_raw is not None else 1.0) * float(offset)
                except Exception:
                    effective_offset = None
        elif scroll_to is not None:
            offset_range = self._quiet_get_slice_offset_range(view=canonical_view)
            if isinstance(offset_range, dict):
                lo = offset_range.get("min_offset")
                hi = offset_range.get("max_offset")
                if effective_axis_label is None:
                    axis_label_raw = offset_range.get("axis_label")
                    if axis_label_raw is not None:
                        effective_axis_label = str(axis_label_raw).strip().upper()
                try:
                    lo_f = float(lo)
                    hi_f = float(hi)
                    t = max(0.0, min(1.0, float(scroll_to)))
                    effective_offset = lo_f + t * (hi_f - lo_f)
                except Exception:
                    effective_offset = None
        if effective_offset is None and allow_viewer_state_fallback:
            state = self._quiet_get_viewer_state()
            if isinstance(state, dict):
                views = state.get("views")
                payload = views.get(canonical_view) if isinstance(views, dict) else None
                if isinstance(payload, dict):
                    if effective_axis_label is None and payload.get("canonical_axis_label") is not None:
                        effective_axis_label = str(payload.get("canonical_axis_label")).strip().upper()
                    canonical_slice_offset = payload.get("canonical_slice_offset")
                    if canonical_slice_offset is not None:
                        try:
                            effective_offset = float(canonical_slice_offset)
                        except Exception:
                            effective_offset = None
        if effective_offset is None and canonical_offset is None and scroll_to is None and offset is not None:
            try:
                effective_offset = float(offset)
            except Exception:
                effective_offset = None
        line = self._format_slice_mm_overlay_line(view=canonical_view, offset_mm=effective_offset, axis_label=effective_axis_label)
        if line:
            return self._stamp_png_corner_text(png_path=png_path, lines=[line])
        return str(png_path)

    def _stamp_current_offsets_on_png(self, *, png_path: str) -> str:
        state = self._quiet_get_viewer_state()
        if not isinstance(state, dict):
            return str(png_path)
        views = state.get("views")
        if not isinstance(views, dict):
            return str(png_path)
        lines = []
        for view_name in _STANDARD_VIEWS:
            payload = views.get(view_name)
            if not isinstance(payload, dict):
                continue
            line = self._format_slice_mm_overlay_line(
                view=view_name,
                offset_mm=payload.get("canonical_slice_offset") if payload.get("canonical_slice_offset") is not None else payload.get("slice_offset"),
                axis_label=payload.get("canonical_axis_label"),
            )
            if line:
                lines.append(line)
        if lines:
            return self._stamp_png_corner_text(png_path=png_path, lines=lines)
        return str(png_path)

    def _remember_view_capture(
        self,
        *,
        view: str,
        png_path: str,
        crop_box: Sequence[int | float],
        full_size: Sequence[int | float],
        raw_png_path: Optional[str] = None,
        orientation: Optional[str] = None,
        offset: Optional[float] = None,
        scroll_to: Optional[float] = None,
    ) -> None:
        canonical_view = _canonical_view(view)
        contexts = dict(self.session.state.get("view_capture_contexts") or {})
        context: Dict[str, Any] = {
            "view": canonical_view,
            "png_path": str(png_path),
            "crop_box": [int(round(float(v))) for v in crop_box],
            "full_size": [int(round(float(v))) for v in full_size],
            "ts": time.time(),
        }
        if raw_png_path:
            context["raw_png_path"] = str(raw_png_path)
        if orientation is not None:
            context["orientation"] = str(orientation)
        if offset is not None:
            context["offset"] = float(offset)
        if scroll_to is not None:
            context["scroll_to"] = float(scroll_to)
        contexts[canonical_view] = context
        self.session.update_state(view_capture_contexts=contexts)

    def _get_view_capture_context(self, *, view: str) -> Optional[Dict[str, Any]]:
        canonical_view = _canonical_view(view)
        contexts = self.session.state.get("view_capture_contexts") or {}
        payload = contexts.get(canonical_view) if isinstance(contexts, dict) else None
        if not isinstance(payload, dict):
            return None
        return dict(payload)

    def capture_slice_view_png(
        self,
        *,
        view: str = "axial",
        out_name: Optional[str] = None,
        include_controller: bool = False,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        canonical_view = _canonical_view(view)
        args = {"view": canonical_view, "include_controller": bool(include_controller), "out_name": out_name, "timeout_s": timeout_s}
        try:
            if out_name is None:
                out_name = f"sliceview_{canonical_view}.png"
            out_path = self._artifact_path(out_name)
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="capture_slice_view_png",
                args={"view": canonical_view, "output_path": str(out_path), "include_controller": bool(include_controller)},
                session_id=self.session.session_id,
                timeout_s=timeout_s,
            )
            saved_path = Path(str((result or {}).get("output_path") or out_path))
            # On some local Slicer/Qt builds the bridge returns before the PNG is visible to the
            # client process. Give the filesystem a brief chance to settle, then retry once.
            if not saved_path.exists():
                for _ in range(5):
                    time.sleep(0.05)
                    if saved_path.exists():
                        break
            if not saved_path.exists():
                result = self.client.exec_bridge(
                    bridge_dir=self.bridge_dir,
                    tool="capture_slice_view_png",
                    args={"view": canonical_view, "output_path": str(saved_path), "include_controller": bool(include_controller)},
                    session_id=self.session.session_id,
                    timeout_s=timeout_s,
                )
                saved_path = Path(str((result or {}).get("output_path") or saved_path))
                if not saved_path.exists():
                    for _ in range(10):
                        time.sleep(0.05)
                        if saved_path.exists():
                            break
            if not saved_path.exists():
                raise FileNotFoundError(f"capture_slice_view_png did not produce output file: {saved_path}")
            cropped_png_path, crop_box, full_size = self._crop_focus_png_in_place(png_path=str(saved_path))
            cropped_png_path = self._stamp_slice_mm_on_png(
                png_path=str(cropped_png_path),
                view=canonical_view,
                offset=((result or {}).get("slice_offset") if isinstance(result, dict) else None),
                canonical_offset=((result or {}).get("canonical_slice_offset") if isinstance(result, dict) else None),
                axis_label=((result or {}).get("canonical_axis_label") if isinstance(result, dict) else None),
                allow_viewer_state_fallback=False,
            )
            if isinstance(result, dict) and result.get("ok") is False:
                raise FileNotFoundError(str(result.get("error") or "capture_slice_view_png bridge returned ok=false"))
            result = {**(result or {}), "png_path": cropped_png_path, "crop_box": list(crop_box), "full_size": list(full_size)}
            self._log(tool="capture_slice_view_png", args=args, result=result, artifacts={"png": str(saved_path)})
            return result
        except Exception as e:
            self._log(tool="capture_slice_view_png", args=args, result=None, ok=False, error=str(e))
            raise

    def _refresh_viewer_state_quiet(self) -> None:
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_viewer_state",
                args={},
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            if isinstance(result, dict) and result.get("ok"):
                self.session.update_state(viewer_state=result)
        except Exception:
            logger.debug("viewer state refresh failed", exc_info=True)

    def _refresh_scene_inventory_quiet(self) -> None:
        try:
            scene_inventory = self.client.list_volumes()
            self.session.update_state(
                scene_inventory=scene_inventory,
                scene_inventory_text=format_scene_inventory_for_prompt(
                    scene_inventory,
                    active_volume_id=self.session.state.get("active_volume_id"),
                    active_volume_name=self.session.state.get("active_volume_name"),
                ),
            )
        except Exception:
            logger.debug("scene inventory refresh failed", exc_info=True)

    def _maybe_refresh_state_before_log(self, tool: str) -> None:
        if tool in _VIEWER_MUTATION_TOOLS:
            self._refresh_viewer_state_quiet()
        if tool in _SCENE_MUTATION_TOOLS:
            self._refresh_scene_inventory_quiet()
