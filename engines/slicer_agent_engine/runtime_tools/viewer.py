
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

from .common import _as_path, _canonical_view, _normalize_public_view_payload, format_scene_inventory_for_prompt, logger

if TYPE_CHECKING:
    from ..algorithm_runtime import AlgorithmRuntime
    from ..gemini_video import GeminiVideoAnalyzer

class ToolContextViewerMixin:
    def get_viewer_state(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_viewer_state",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            if result.get("ok"):
                # Store the last known full viewer state for debugging/replay.
                self.session.update_state(viewer_state=result)
            self._log(tool="get_viewer_state", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="get_viewer_state", args=args, result=None, ok=False, error=str(e))
            raise

    def set_window_level(
        self,
        *,
        window: Optional[float] = None,
        level: Optional[float] = None,
        auto: Optional[bool] = None,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "window": window,
            "level": level,
            "auto": auto,
            "volume_id": volume_id,
            "volume_name": volume_name,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_window_level",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_window_level", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_window_level", args=args, result=None, ok=False, error=str(e))
            raise

    def apply_window_preset(
        self,
        *,
        preset: str,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply a clinically common window preset.

        Supported presets:
          - ct_lung / lung:          W=1500, L=-600
          - ct_mediastinal / mediastinal / soft_tissue: W=400, L=40
          - ct_bone / bone:          W=2000, L=300
          - pet / pet_auto / auto:   auto window/level
        """
        key = str(preset or '').strip().lower()
        mapping = {
            'ct_lung': (1500.0, -600.0),
            'lung': (1500.0, -600.0),
            'ct_mediastinal': (400.0, 40.0),
            'mediastinal': (400.0, 40.0),
            'soft_tissue': (400.0, 40.0),
            'ct_soft_tissue': (400.0, 40.0),
            'ct_bone': (2000.0, 300.0),
            'bone': (2000.0, 300.0),
        }
        if key in {'pet', 'pet_auto', 'auto', 'pet_default'}:
            result = self.set_window_level(auto=True, volume_id=volume_id, volume_name=volume_name)
            self._log(tool='apply_window_preset', args={'preset': preset, 'volume_id': volume_id, 'volume_name': volume_name}, result=result)
            return result
        if key not in mapping:
            raise ValueError(f'Unsupported window preset: {preset}')
        window, level = mapping[key]
        result = self.set_window_level(window=window, level=level, auto=False, volume_id=volume_id, volume_name=volume_name)
        self._log(tool='apply_window_preset', args={'preset': preset, 'volume_id': volume_id, 'volume_name': volume_name}, result=result)
        return result

    def set_interpolation(
        self,
        *,
        interpolate: bool,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {"interpolate": bool(interpolate), "volume_id": volume_id, "volume_name": volume_name}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_interpolation",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_interpolation", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_interpolation", args=args, result=None, ok=False, error=str(e))
            raise

    def set_slice_orientation(self, *, view: str, orientation: str) -> Dict[str, Any]:
        args = {"view": view, "orientation": orientation}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_slice_orientation",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_slice_orientation", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_slice_orientation", args=args, result=None, ok=False, error=str(e))
            raise

    def get_slice_offset_range(self, *, view: str) -> Dict[str, Any]:
        args = {"view": view}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="get_slice_offset_range",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="get_slice_offset_range", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="get_slice_offset_range", args=args, result=None, ok=False, error=str(e))
            raise

    def set_slice_offset(self, *, view: str, offset: float) -> Dict[str, Any]:
        args = {"view": view, "offset": float(offset)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_slice_offset",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_slice_offset", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_slice_offset", args=args, result=None, ok=False, error=str(e))
            raise

    def set_slice_scroll_to(self, *, view: str, scroll_to: float) -> Dict[str, Any]:
        args = {"view": view, "scroll_to": float(scroll_to)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_slice_scroll_to",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_slice_scroll_to", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_slice_scroll_to", args=args, result=None, ok=False, error=str(e))
            raise

    def fit_slice(self, *, view: str) -> Dict[str, Any]:
        args = {"view": view}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="fit_slice",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="fit_slice", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="fit_slice", args=args, result=None, ok=False, error=str(e))
            raise

    def zoom_slice_relative(self, *, view: str, factor: float) -> Dict[str, Any]:
        args = {"view": view, "factor": float(factor)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="zoom_slice_relative",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="zoom_slice_relative", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="zoom_slice_relative", args=args, result=None, ok=False, error=str(e))
            raise

    def set_field_of_view(self, *, view: str, field_of_view: Sequence[float]) -> Dict[str, Any]:
        args = {"view": view, "field_of_view": [float(v) for v in field_of_view]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_field_of_view",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_field_of_view", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_field_of_view", args=args, result=None, ok=False, error=str(e))
            raise

    def set_xyz_origin(self, *, view: str, xyz_origin: Sequence[float]) -> Dict[str, Any]:
        args = {"view": view, "xyz_origin": [float(v) for v in xyz_origin]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_xyz_origin",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_xyz_origin", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_xyz_origin", args=args, result=None, ok=False, error=str(e))
            raise

    def jump_to_ras(self, *, ras: Sequence[float], centered: bool = True) -> Dict[str, Any]:
        args = {"ras": [float(v) for v in ras], "centered": bool(centered)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="jump_to_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="jump_to_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="jump_to_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def recover_standard_views(
        self,
        *,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
        centered: bool = True,
        fit: bool = True,
    ) -> Dict[str, Any]:
        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "centered": bool(centered),
            "fit": bool(fit),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="recover_standard_views",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="recover_standard_views", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="recover_standard_views", args=args, result=None, ok=False, error=str(e))
            raise

    def set_linked_slices(self, *, enabled: bool) -> Dict[str, Any]:
        args = {"enabled": bool(enabled)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_linked_slices",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_linked_slices", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_linked_slices", args=args, result=None, ok=False, error=str(e))
            raise

    def set_layout(self, *, layout: str) -> Dict[str, Any]:
        args = {"layout": layout}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_layout",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_layout", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_layout", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Rendering (PNG) + cine (MP4)
    # -------------------------

    def get_slice_png(
        self,
        *,
        view: str = "axial",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
        out_name: Optional[str] = None,
        offset: Optional[float] = None,
        copy_slice_geometry_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        canonical_view = _canonical_view(view)
        args = {
            "view": canonical_view,
            "orientation": orientation,
            "scroll_to": float(scroll_to),
            "size": int(size),
            "offset": offset,
            "copy_slice_geometry_from": copy_slice_geometry_from,
        }
        try:
            if out_name is None:
                out_name = f"slice_{canonical_view}_{orientation}_{scroll_to:.3f}_{int(size)}.png"
            out_path = self._artifact_path(out_name)
            self.client.save_slice_png(
                out_path,
                view=canonical_view,
                orientation=orientation,
                scroll_to=scroll_to,
                size=size,
                offset=offset,
                copy_slice_geometry_from=copy_slice_geometry_from,
            )
            cropped_png_path, crop_box, full_size = self._crop_focus_png_in_place(png_path=str(out_path))
            cropped_png_path = self._stamp_slice_mm_on_png(
                png_path=str(cropped_png_path),
                view=canonical_view,
                offset=(float(offset) if offset is not None else None),
                scroll_to=(float(scroll_to) if offset is None else None),
            )
            self._remember_view_capture(
                view=canonical_view,
                png_path=cropped_png_path,
                raw_png_path=str(out_path),
                crop_box=crop_box,
                full_size=full_size,
                orientation=str(orientation),
                offset=(float(offset) if offset is not None else None),
                scroll_to=float(scroll_to),
            )
            result = {"png_path": cropped_png_path, "crop_box": list(crop_box), "full_size": list(full_size)}
            self._log(tool="get_slice_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_slice_png", args=args, result=None, ok=False, error=str(e))
            raise

    def get_screenshot_png(self, *, out_name: str = "screenshot.png") -> Dict[str, Any]:
        args = {"out_name": out_name}
        try:
            out_path = self._artifact_path(out_name)
            self.client.save_screenshot_png(out_path)
            self._stamp_current_offsets_on_png(png_path=str(out_path))
            result = {"png_path": str(out_path)}
            self._log(tool="get_screenshot_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_screenshot_png", args=args, result=None, ok=False, error=str(e))
            raise

    def get_threeD_png(self, *, look_from_axis: Optional[str] = None, out_name: str = "threeD.png") -> Dict[str, Any]:
        args = {"look_from_axis": look_from_axis, "out_name": out_name}
        try:
            out_path = self._artifact_path(out_name)
            self.client.save_threeD_png(out_path, look_from_axis=look_from_axis)
            self._stamp_current_offsets_on_png(png_path=str(out_path))
            result = {"png_path": str(out_path)}
            self._log(tool="get_threeD_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_threeD_png", args=args, result=None, ok=False, error=str(e))
            raise

    def get_timeimage_png(self, *, color: str = "pink", out_name: str = "timeimage.png") -> Dict[str, Any]:
        args = {"color": color, "out_name": out_name}
        try:
            out_path = self._artifact_path(out_name)
            self.client.save_timeimage_png(out_path, color=color)
            self._stamp_current_offsets_on_png(png_path=str(out_path))
            result = {"png_path": str(out_path)}
            self._log(tool="get_timeimage_png", args=args, result=result, artifacts={"png": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="get_timeimage_png", args=args, result=None, ok=False, error=str(e))
            raise

    def capture_cine(
        self,
        *,
        view: str = "axial",
        orientation: str = "axial",
        start: float = 0.0,
        end: float = 1.0,
        n_frames: int = 60,
        size: int = 512,
        fps: int = 12,
        frames_dir_name: Optional[str] = None,
        out_mp4_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.video is None:
            raise RuntimeError("VideoRenderer is not configured (ffmpeg not available)")

        canonical_view = _canonical_view(view)

        args = {
            "view": canonical_view,
            "orientation": orientation,
            "start": float(start),
            "end": float(end),
            "n_frames": int(n_frames),
            "size": int(size),
            "fps": int(fps),
        }

        try:
            if frames_dir_name is None:
                frames_dir_name = f"frames_{canonical_view}_{orientation}_{n_frames}f"
            frames_dir = self._artifact_path(frames_dir_name)
            frames_dir.mkdir(parents=True, exist_ok=True)

            if out_mp4_name is None:
                out_mp4_name = f"cine_{canonical_view}_{orientation}_{n_frames}f_{fps}fps.mp4"
            out_mp4 = self._artifact_path(out_mp4_name)

            if n_frames <= 1:
                raise ValueError("n_frames must be >= 2")

            for i in range(n_frames):
                t = start + (end - start) * (i / (n_frames - 1))
                frame_path = frames_dir / f"frame_{i:05d}.png"
                self.client.save_slice_png(frame_path, view=canonical_view, orientation=orientation, scroll_to=float(t), size=size)

            mp4_path = self.video.encode_mp4_from_pattern(
                input_pattern=frames_dir / "frame_%05d.png",
                output_path=out_mp4,
                fps=fps,
            )

            result = {"mp4_path": str(mp4_path), "frames_dir": str(frames_dir)}
            self._log(tool="capture_cine", args=args, result=result, artifacts={"mp4": str(mp4_path)})
            return result

        except Exception as e:
            self._log(tool="capture_cine", args=args, result=None, ok=False, error=str(e))
            raise
