
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

from .common import _STANDARD_VIEWS, _as_path, _canonical_view, _normalize_public_view_payload, format_scene_inventory_for_prompt, logger
from ..click_contract import bbox_1000_to_full_norm, bbox_center_norm, crop_png_to_focus, normalize_bbox_1000, normalize_point_1000, point_1000_to_full_norm, point_norm_from_1000

if TYPE_CHECKING:
    from ..algorithm_runtime import AlgorithmRuntime
    from ..gemini_video import GeminiVideoAnalyzer

class ToolContextSegmentationMixin:
    def _apply_segmentation_state_from_result(self, result: Dict[str, Any]) -> None:
        updates: Dict[str, Any] = {}
        if result.get("segmentation_id") is not None:
            updates["active_segmentation_id"] = result.get("segmentation_id")
        if result.get("segmentation_name") is not None:
            updates["active_segmentation_name"] = result.get("segmentation_name")
        if result.get("segment_id") is not None:
            updates["active_segment_id"] = result.get("segment_id")
        if result.get("segment_name") is not None:
            updates["active_segment_name"] = result.get("segment_name")
        if result.get("source_volume_id") is not None:
            updates["active_volume_id"] = result.get("source_volume_id")
        if result.get("source_volume_name") is not None:
            updates["active_volume_name"] = result.get("source_volume_name")
        if updates:
            self.session.update_state(**updates)

    def _resolve_view_capture_reference(self, *, view: str, size: int, purpose: str) -> Dict[str, Any]:
        canonical_view = _canonical_view(view)
        capture_ctx = self._get_view_capture_context(view=canonical_view)
        if isinstance(capture_ctx, dict):
            png_path = capture_ctx.get("png_path")
            crop_box = capture_ctx.get("crop_box")
            full_size = capture_ctx.get("full_size")
            if png_path and crop_box and full_size and Path(str(png_path)).exists():
                return {
                    "png_path": str(png_path),
                    "crop_box": tuple(int(v) for v in crop_box),
                    "full_size": tuple(int(v) for v in full_size),
                    "orientation": capture_ctx.get("orientation"),
                    "offset": capture_ctx.get("offset"),
                    "scroll_to": capture_ctx.get("scroll_to"),
                    "source": "session_view_capture",
                }

        raw_capture = self._capture_current_view_png_raw(
            view=canonical_view,
            size=int(size),
            out_name=f"{purpose}_raw_{canonical_view}.png",
        )
        cropped_path = self._artifact_path(f"{purpose}_{canonical_view}.png")
        _, crop_box, full_size = crop_png_to_focus(str(raw_capture["png_path"]), out_path=cropped_path)
        orientation = raw_capture.get("orientation")
        offset = raw_capture.get("offset")
        self._stamp_slice_mm_on_png(
            png_path=str(cropped_path),
            view=canonical_view,
            offset=(float(offset) if offset is not None else None),
            canonical_offset=(float(raw_capture["canonical_offset"]) if raw_capture.get("canonical_offset") is not None else None),
            axis_label=raw_capture.get("canonical_axis_label"),
        )
        self._remember_view_capture(
            view=canonical_view,
            png_path=str(cropped_path),
            raw_png_path=str(raw_capture["png_path"]),
            crop_box=crop_box,
            full_size=full_size,
            orientation=(str(orientation) if orientation is not None else None),
            offset=(float(offset) if offset is not None else None),
        )
        return {
            "png_path": str(cropped_path),
            "crop_box": tuple(int(v) for v in crop_box),
            "full_size": tuple(int(v) for v in full_size),
            "orientation": orientation,
            "offset": offset,
            "scroll_to": None,
            "source": "fresh_capture",
        }

    def list_segmentations(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="list_segmentations",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict) and result.get("segmentations"):
                active_seg = result.get("active_segmentation") or {}
                if isinstance(active_seg, dict):
                    self._apply_segmentation_state_from_result(
                        {
                            "segmentation_id": active_seg.get("id"),
                            "segmentation_name": active_seg.get("name"),
                        }
                    )
            self._log(tool="list_segmentations", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_segmentations", args=args, result=None, ok=False, error=str(e))
            raise

    def list_segments(
        self,
        *,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="list_segments",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="list_segments", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_segments", args=args, result=None, ok=False, error=str(e))
            raise

    def show_segmentation(
        self,
        *,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
        visible: bool = True,
        fill: bool = True,
        outline: bool = True,
        opacity: float = 0.3,
        color: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        args = {
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
            "visible": bool(visible),
            "fill": bool(fill),
            "outline": bool(outline),
            "opacity": float(opacity),
            "color": list(color) if color is not None else None,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="show_segmentation",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
                artifact_lookup = dict(self.session.state.get("vista3d_segmentation_artifacts") or {})
                selected_artifacts = None
                seg_id = result.get("segmentation_id") or args.get("segmentation_id")
                seg_name = result.get("segmentation_name") or args.get("segmentation_name")
                if seg_id and str(seg_id) in artifact_lookup:
                    selected_artifacts = artifact_lookup.get(str(seg_id))
                if selected_artifacts is None and seg_name:
                    for item in artifact_lookup.values():
                        if isinstance(item, dict) and item.get("segmentation_name") == seg_name:
                            selected_artifacts = item
                            break
                if isinstance(selected_artifacts, dict) and selected_artifacts.get("labelmap_volume_id"):
                    try:
                        result["labelmap_overlay"] = self.client.exec_bridge(
                            bridge_dir=self.bridge_dir,
                            tool="set_labelmap_overlay",
                            args={
                                "labelmap_volume_id": selected_artifacts.get("labelmap_volume_id"),
                                "labelmap_volume_name": selected_artifacts.get("labelmap_volume_name"),
                                "reference_volume_id": selected_artifacts.get("source_volume_id") or self.session.state.get("active_volume_id"),
                                "reference_volume_name": selected_artifacts.get("source_volume_name") or self.session.state.get("active_volume_name"),
                                "visible": bool(visible),
                                "opacity": float(opacity),
                                "visible_label_values": selected_artifacts.get("label_values"),
                            },
                            session_id=self.session.session_id,
                        )
                    except Exception as overlay_error:
                        result["labelmap_overlay_error"] = str(overlay_error)
            self._log(tool="show_segmentation", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="show_segmentation", args=args, result=None, ok=False, error=str(e))
            raise

    def hide_segmentation(
        self,
        *,
        target: str = "all",
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "target": str(target or "all"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id,
            "segment_name": segment_name,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="hide_segmentation",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="hide_segmentation", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="hide_segmentation", args=args, result=None, ok=False, error=str(e))
            raise

    def image_coords_to_ras(
        self,
        *,
        view: str,
        bbox_1000: Optional[Sequence[int | float]] = None,
        point_1000: Optional[Sequence[int | float]] = None,
        size: int = 512,
    ) -> Dict[str, Any]:
        def _is_zero_bbox(candidate: Optional[Sequence[int | float]]) -> bool:
            if candidate is None:
                return False
            try:
                values = [float(v) for v in candidate]
            except Exception:
                return False
            return len(values) == 4 and all(v == 0.0 for v in values)

        def _is_zero_point(candidate: Optional[Sequence[int | float]]) -> bool:
            if candidate is None:
                return False
            try:
                values = [float(v) for v in candidate]
            except Exception:
                return False
            return len(values) == 2 and all(v == 0.0 for v in values)

        def _sanitize_dual_inputs(
            raw_bbox: Optional[Sequence[int | float]],
            raw_point: Optional[Sequence[int | float]],
        ) -> tuple[Optional[Sequence[int | float]], Optional[Sequence[int | float]], Optional[str]]:
            if raw_bbox is None or raw_point is None:
                return raw_bbox, raw_point, None
            if _is_zero_point(raw_point) and not _is_zero_bbox(raw_bbox):
                return raw_bbox, None, "dropped_zero_point_placeholder"
            if _is_zero_bbox(raw_bbox) and not _is_zero_point(raw_point):
                return None, raw_point, "dropped_zero_bbox_placeholder"
            try:
                norm_bbox_local = normalize_bbox_1000(raw_bbox)
                norm_point_local = normalize_point_1000(raw_point)
            except Exception:
                return raw_bbox, raw_point, None

            px, py = int(norm_point_local[0]), int(norm_point_local[1])
            point_inside_bbox = (
                int(norm_bbox_local[0]) <= px <= int(norm_bbox_local[2])
                and int(norm_bbox_local[1]) <= py <= int(norm_bbox_local[3])
            )
            return None, norm_point_local, (
                "dropped_bbox_1000_point_preferred_inside_bbox"
                if point_inside_bbox
                else "dropped_bbox_1000_point_preferred_outside_bbox"
            )

        original_bbox_1000 = list(bbox_1000) if bbox_1000 is not None else None
        original_point_1000 = list(point_1000) if point_1000 is not None else None
        contract_fix_applied: Optional[str] = None

        if bbox_1000 is not None and len(bbox_1000) == 0:
            bbox_1000 = None
        if point_1000 is not None and len(point_1000) == 0:
            point_1000 = None
        if bbox_1000 is not None and point_1000 is not None:
            bbox_1000, point_1000, contract_fix_applied = _sanitize_dual_inputs(bbox_1000, point_1000)
            if bbox_1000 is not None and point_1000 is not None:
                raise ValueError(
                    "Provide exactly one of bbox_1000 or point_1000. Never send both. "
                    "If you want the center of a bbox, send only bbox_1000 and use center_ras from the result. "
                    "If you want a single clicked point, send only point_1000."
                )

        canonical_view = _canonical_view(view)
        if (bbox_1000 is None) == (point_1000 is None):
            raise ValueError("Provide exactly one of bbox_1000 or point_1000. Never send both. If you want the center of a bbox, send only bbox_1000 and use center_ras from the result.")

        norm_bbox: Optional[List[int]] = None
        norm_point: Optional[List[int]] = None
        point_norm: Optional[List[float]] = None
        mapped_bbox_norm: Optional[List[float]] = None
        mapped_point_norm: Optional[List[float]] = None
        preview_png_path: Optional[str] = None
        crop_box: Optional[tuple[int, int, int, int]] = None
        full_size: Optional[tuple[int, int]] = None
        capture_ref: Optional[Dict[str, Any]] = None
        bridge_pose: Dict[str, Any] = {}

        try:
            capture_ref = self._resolve_view_capture_reference(view=canonical_view, size=int(size), purpose="coords")
            base_png = str(capture_ref.get("png_path")) if capture_ref.get("png_path") else None
            crop_box = tuple(capture_ref.get("crop_box")) if capture_ref.get("crop_box") else None
            full_size = tuple(capture_ref.get("full_size")) if capture_ref.get("full_size") else None
            if capture_ref.get("orientation") is not None:
                bridge_pose["orientation"] = str(capture_ref.get("orientation"))
            if capture_ref.get("offset") is not None:
                bridge_pose["offset"] = float(capture_ref.get("offset"))
            elif capture_ref.get("scroll_to") is not None:
                bridge_pose["scroll_to"] = float(capture_ref.get("scroll_to"))
        except Exception:
            base_png = None
            crop_box = None
            full_size = None
            capture_ref = None
            bridge_pose = {}

        if bbox_1000 is not None:
            norm_bbox = normalize_bbox_1000(bbox_1000)
            cx_norm, cy_norm = bbox_center_norm(norm_bbox)
            point_norm = [float(cx_norm), float(cy_norm)]
            if crop_box is not None and full_size is not None:
                mapped_bbox = bbox_1000_to_full_norm(norm_bbox, crop_box=crop_box, full_width=full_size[0], full_height=full_size[1])
                mapped_bbox_norm = [float(v) for v in mapped_bbox]
                mapped_point_norm = [float((mapped_bbox[0] + mapped_bbox[2]) / 2.0), float((mapped_bbox[1] + mapped_bbox[3]) / 2.0)]
                bridge_args = {"view": canonical_view, "bbox_norm": mapped_bbox_norm, **bridge_pose}
            else:
                mapped_point_norm = point_norm
                bridge_args = {"view": canonical_view, "bbox_norm": [float(norm_bbox[0]) / 999.0, float(norm_bbox[1]) / 999.0, float(norm_bbox[2]) / 999.0, float(norm_bbox[3]) / 999.0], **bridge_pose}
            if base_png:
                try:
                    preview_png_path = self._annotate_bbox_preview(
                        png_path=base_png,
                        bbox_1000=norm_bbox,
                        out_name=f"coords_{canonical_view}_bbox_annotated.png",
                        label="bbox1000",
                    )
                except Exception:
                    logger.exception("Failed annotating bbox preview", exc_info=True)
        else:
            norm_point = normalize_point_1000(point_1000 or [])
            px_norm, py_norm = point_norm_from_1000(norm_point)
            point_norm = [float(px_norm), float(py_norm)]
            if crop_box is not None and full_size is not None:
                mx, my = point_1000_to_full_norm(norm_point, crop_box=crop_box, full_width=full_size[0], full_height=full_size[1])
                mapped_point_norm = [float(mx), float(my)]
                bridge_args = {"view": canonical_view, "point_norm": mapped_point_norm, **bridge_pose}
            else:
                mapped_point_norm = point_norm
                bridge_args = {"view": canonical_view, "point_norm": [float(px_norm), float(py_norm)], **bridge_pose}
            if base_png:
                try:
                    preview_png_path = self._annotate_point_preview(
                        png_path=base_png,
                        point_1000=norm_point,
                        out_name=f"coords_{canonical_view}_point_annotated.png",
                        label="point1000",
                    )
                except Exception:
                    logger.exception("Failed annotating point preview", exc_info=True)

        args = {
            "view": canonical_view,
            "bbox_1000": norm_bbox,
            "point_1000": norm_point,
            "size": int(size),
            "original_bbox_1000": original_bbox_1000,
            "original_point_1000": original_point_1000,
            "contract_fix_applied": contract_fix_applied,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="image_coords_to_ras",
                args=bridge_args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                if norm_bbox is not None:
                    result["bbox_1000"] = norm_bbox
                if norm_point is not None:
                    result["point_1000"] = norm_point
                if point_norm is not None:
                    result["point_norm"] = point_norm
                if contract_fix_applied:
                    result["contract_fix_applied"] = contract_fix_applied
                    result["original_bbox_1000"] = original_bbox_1000
                    result["original_point_1000"] = original_point_1000
                    result["effective_input_kind"] = "bbox_1000" if norm_bbox is not None else "point_1000"
                if mapped_bbox_norm is not None:
                    result["mapped_bbox_norm"] = mapped_bbox_norm
                if mapped_point_norm is not None:
                    result["mapped_point_norm"] = mapped_point_norm
                if crop_box is not None:
                    result["crop_box"] = list(crop_box)
                if full_size is not None:
                    result["full_size"] = list(full_size)
                if preview_png_path:
                    result["png_path"] = preview_png_path
                if capture_ref is not None:
                    result["reference_png_path"] = capture_ref.get("png_path")
                    result["reference_source"] = capture_ref.get("source")
            self._log(tool="image_coords_to_ras", args=args, result=result, artifacts={"png": preview_png_path} if preview_png_path else None)
            return result
        except Exception as e:
            self._log(tool="image_coords_to_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_from_threshold(
        self,
        *,
        minimum_threshold: float,
        maximum_threshold: float,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "minimum_threshold": float(minimum_threshold),
            "maximum_threshold": float(maximum_threshold),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_from_threshold",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_from_threshold", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_from_threshold", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_local_threshold(
        self,
        *,
        minimum_threshold: float,
        maximum_threshold: float,
        minimum_diameter_mm: float = 3.0,
        feature_size_mm: float = 3.0,
        segmentation_algorithm: str = "GrowCut",
        seed_view: Optional[str] = None,
        seed_bbox_1000: Optional[Sequence[int | float]] = None,
        seed_x_norm: Optional[float] = None,
        seed_y_norm: Optional[float] = None,
        ras_seed: Optional[Sequence[float]] = None,
        seed_segment_id: Optional[str] = None,
        seed_segment_name: Optional[str] = None,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
        size: int = 512,
    ) -> Dict[str, Any]:
        bbox_norm: Optional[List[int]] = None
        seed_preview_png_path: Optional[str] = None
        seed_crop_box: Optional[tuple[int, int, int, int]] = None
        seed_full_size: Optional[tuple[int, int]] = None
        seed_capture_ref: Optional[Dict[str, Any]] = None
        seed_orientation: Optional[str] = None
        seed_offset: Optional[float] = None
        seed_scroll_to: Optional[float] = None
        if seed_bbox_1000 is None and (seed_x_norm is None or seed_y_norm is None) and ras_seed is None and not (seed_segment_id or seed_segment_name):
            raise ValueError("Provide seed_bbox_1000 + seed_view (preferred), or seed_x_norm/seed_y_norm + seed_view, or ras_seed, or seed_segment_id/seed_segment_name.")
        if seed_bbox_1000 is not None:
            bbox_norm = normalize_bbox_1000(seed_bbox_1000)
            if seed_view is None:
                raise ValueError("seed_view is required when using seed_bbox_1000")
            center_1000 = [int(round((bbox_norm[0] + bbox_norm[2]) / 2.0)), int(round((bbox_norm[1] + bbox_norm[3]) / 2.0))]
            try:
                seed_capture_ref = self._resolve_view_capture_reference(
                    view=str(seed_view),
                    size=int(size),
                    purpose="segment_local_threshold_seed",
                )
                seed_crop_box = tuple(seed_capture_ref.get("crop_box")) if seed_capture_ref.get("crop_box") else None
                seed_full_size = tuple(seed_capture_ref.get("full_size")) if seed_capture_ref.get("full_size") else None
                seed_orientation = str(seed_capture_ref.get("orientation")) if seed_capture_ref.get("orientation") is not None else None
                seed_offset = float(seed_capture_ref.get("offset")) if seed_capture_ref.get("offset") is not None else None
                seed_scroll_to = float(seed_capture_ref.get("scroll_to")) if seed_capture_ref.get("scroll_to") is not None else None
                if seed_full_size is not None and seed_crop_box is not None:
                    seed_x_norm, seed_y_norm = point_1000_to_full_norm(center_1000, crop_box=seed_crop_box, full_width=seed_full_size[0], full_height=seed_full_size[1])
                else:
                    seed_x_norm, seed_y_norm = bbox_center_norm(bbox_norm)
                if seed_capture_ref.get("png_path"):
                    seed_preview_png_path = self._annotate_bbox_preview(
                        png_path=str(seed_capture_ref["png_path"]),
                        bbox_1000=bbox_norm,
                        out_name=f"segment_local_threshold_seed_{_canonical_view(seed_view)}_annotated.png",
                        label=None,
                    )
            except Exception:
                seed_x_norm, seed_y_norm = bbox_center_norm(bbox_norm)
                logger.exception("Failed to create local-threshold seed preview", exc_info=True)

        args = {
            "minimum_threshold": float(minimum_threshold),
            "maximum_threshold": float(maximum_threshold),
            "minimum_diameter_mm": float(minimum_diameter_mm),
            "feature_size_mm": float(feature_size_mm),
            "segmentation_algorithm": str(segmentation_algorithm),
            "seed_view": seed_view,
            "seed_bbox_1000": bbox_norm,
            "seed_x_norm": (float(seed_x_norm) if seed_x_norm is not None else None),
            "seed_y_norm": (float(seed_y_norm) if seed_y_norm is not None else None),
            "ras_seed": ([float(v) for v in ras_seed] if ras_seed is not None else None),
            "seed_segment_id": seed_segment_id,
            "seed_segment_name": seed_segment_name,
            "seed_orientation": seed_orientation,
            "seed_offset_mm": seed_offset,
            "seed_scroll_to": seed_scroll_to,
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
            "size": int(size),
        }
        bridge_args = {k: v for k, v in args.items() if k not in {"seed_bbox_1000", "size"} and v is not None}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_local_threshold",
                args=bridge_args,
                session_id=self.session.session_id,
            )
            overlay_png_paths: Dict[str, str] = {}
            overlay_warnings: list[str] = []

            zero_voxel_failure = False
            voxel_count: Optional[int] = None
            if isinstance(result, dict) and result.get("voxel_count") is not None:
                try:
                    voxel_count = int(result.get("voxel_count"))
                except Exception:
                    voxel_count = None
                if voxel_count is not None and voxel_count <= 0:
                    zero_voxel_failure = True

            if not zero_voxel_failure:
                try:
                    seed_view_key = _canonical_view(str(seed_view)) if seed_view is not None else "axial"
                    ordered_views = [seed_view_key] + [v for v in _STANDARD_VIEWS if v != seed_view_key]
                    for v in ordered_views:
                        try:
                            cap = self.capture_slice_view_png(
                                view=v,
                                out_name=f"segment_local_threshold_{v}_overlay.png",
                            )
                        except Exception as exc:
                            msg = f"Failed to capture local-threshold overlay preview for view={v}: {exc}"
                            overlay_warnings.append(msg)
                            logger.exception("Failed to capture local-threshold overlay preview for view=%s", v, exc_info=True)
                            continue
                        if isinstance(cap, dict) and cap.get("ok", True) is not False and cap.get("png_path"):
                            candidate_png = str(cap["png_path"])
                            overlay_png_paths[v] = candidate_png
                    if bbox_norm is not None and seed_view is not None and seed_view_key in overlay_png_paths:
                        try:
                            if Path(str(overlay_png_paths[seed_view_key])).exists():
                                overlay_with_seed = self._annotate_bbox_preview(
                                    png_path=overlay_png_paths[seed_view_key],
                                    bbox_1000=bbox_norm,
                                    out_name=f"segment_local_threshold_{seed_view_key}_overlay_seed.png",
                                    label=None,
                                )
                                overlay_png_paths[seed_view_key] = overlay_with_seed
                        except Exception as exc:
                            overlay_warnings.append(f"Failed annotating local-threshold seed overlay: {exc}")
                            logger.exception("Failed annotating local-threshold seed overlay", exc_info=True)
                except Exception as exc:
                    overlay_warnings.append(f"Failed to capture local-threshold overlay previews: {exc}")
                    logger.exception("Failed to capture local-threshold overlay previews", exc_info=True)

            if isinstance(result, dict):
                png_paths: Dict[str, str] = dict(overlay_png_paths)
                if seed_preview_png_path and zero_voxel_failure:
                    png_paths["seed_preview"] = seed_preview_png_path
                result["seed_bbox_1000"] = bbox_norm
                result["seed_center_norm"] = [float(seed_x_norm), float(seed_y_norm)] if seed_x_norm is not None and seed_y_norm is not None else None
                if seed_crop_box is not None:
                    result["seed_crop_box"] = list(seed_crop_box)
                if seed_full_size is not None:
                    result["seed_full_size"] = list(seed_full_size)
                result["seed_preview_png_path"] = seed_preview_png_path
                if seed_capture_ref is not None:
                    result["seed_reference_png_path"] = seed_capture_ref.get("png_path")
                    result["seed_reference_source"] = seed_capture_ref.get("source")
                result["overlay_capture_ok"] = bool(overlay_png_paths)
                if overlay_warnings:
                    result["overlay_capture_warnings"] = overlay_warnings
                if png_paths:
                    result["png_paths"] = png_paths
                    seed_view_key = _canonical_view(str(seed_view)) if seed_view is not None else "axial"
                    if seed_view_key in overlay_png_paths:
                        preferred_png = overlay_png_paths[seed_view_key]
                    elif "axial" in overlay_png_paths:
                        preferred_png = overlay_png_paths["axial"]
                    elif overlay_png_paths:
                        preferred_png = next(iter(overlay_png_paths.values()))
                    else:
                        preferred_png = seed_preview_png_path
                    result["png_path"] = preferred_png
                if zero_voxel_failure:
                    result["ok"] = False
                    result["tool"] = "segment_local_threshold"
                    result["error"] = (
                        "Local-threshold segmentation produced 0 voxels. No usable mask was created. "
                        "Inspect the seed preview, then retry with a tighter seed_bbox_1000 and/or adjusted thresholds."
                    )
                    result["retryable"] = True
                    result["failure_reason"] = "zero_voxels"
                self._apply_segmentation_state_from_result(result)
            artifact_png = result.get("png_path") if isinstance(result, dict) else None
            self._log(tool="segment_local_threshold", args=args, result=result, artifacts={"png": str(artifact_png or seed_preview_png_path or "")})
            return result
        except Exception as e:
            self._log(tool="segment_local_threshold", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_edit_sphere(
        self,
        *,
        ras_center: Sequence[float],
        radius_mm: float,
        action: str = "add",
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "ras_center": [float(v) for v in ras_center],
            "radius_mm": float(radius_mm),
            "action": str(action),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_edit_sphere",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_edit_sphere", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_edit_sphere", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_margin(
        self,
        *,
        margin_size_mm: float,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "margin_size_mm": float(margin_size_mm),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_margin",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_margin", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_margin", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_smoothing(
        self,
        *,
        smoothing_method: str = "MEDIAN",
        kernel_size_mm: float = 3.0,
        gaussian_std_mm: Optional[float] = None,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "smoothing_method": str(smoothing_method),
            "kernel_size_mm": float(kernel_size_mm),
            "gaussian_std_mm": (float(gaussian_std_mm) if gaussian_std_mm is not None else None),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_smoothing",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_smoothing", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_smoothing", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_islands(
        self,
        *,
        operation: str = "KEEP_LARGEST_ISLAND",
        minimum_size: int = 1000,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "operation": str(operation),
            "minimum_size": int(minimum_size),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_islands",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_islands", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_islands", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_logical(
        self,
        *,
        operation: str,
        modifier_segment_id: Optional[str] = None,
        modifier_segment_name: Optional[str] = None,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "operation": str(operation),
            "modifier_segment_id": modifier_segment_id,
            "modifier_segment_name": modifier_segment_name,
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_logical",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_logical", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_logical", args=args, result=None, ok=False, error=str(e))
            raise

    def segment_statistics(
        self,
        *,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="segment_statistics",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="segment_statistics", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="segment_statistics", args=args, result=None, ok=False, error=str(e))
            raise

    def center_on_segment(
        self,
        *,
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="center_on_segment",
                args=args,
                session_id=self.session.session_id,
            )
            if isinstance(result, dict):
                self._apply_segmentation_state_from_result(result)
            self._log(tool="center_on_segment", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="center_on_segment", args=args, result=None, ok=False, error=str(e))
            raise
