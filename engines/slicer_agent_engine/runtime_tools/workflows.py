
from __future__ import annotations

import difflib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

from medflow_prompts import render_prompt

from .common import _STANDARD_VIEWS, _as_path, _canonical_view, _normalize_public_view_payload, format_scene_inventory_for_prompt, logger
from ..slicer_client import SlicerRequestError

if TYPE_CHECKING:
    from ..algorithm_runtime import AlgorithmRuntime
    from ..gemini_video import GeminiVideoAnalyzer


RUNTIME_PROMPT_REF = "3d_slicer/runtime.json"


class ToolContextWorkflowMixin:
    def _reset_registration_preview_to_initial_slice_enabled(self) -> bool:
        raw = os.environ.get("SLICER_RESET_REGISTRATION_PREVIEW_TO_INITIAL_SLICE", "1")
        return str(raw).strip().lower() not in {"0", "false", "no", "off"}

    def _record_initial_mid_slice_views(self) -> Dict[str, Any]:
        """Store the case-opening mid-slice offsets for comparable registration previews."""
        views: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}
        for view in _STANDARD_VIEWS:
            view_payload: Dict[str, Any] = {"view": view, "orientation": view, "scroll_to": 0.5}
            try:
                view_payload["orientation_result"] = self.set_slice_orientation(view=view, orientation=view)
                scroll_result = self.set_slice_scroll_to(view=view, scroll_to=0.5)
                view_payload["scroll_result"] = scroll_result
                if isinstance(scroll_result, dict):
                    if scroll_result.get("slice_offset") is not None:
                        view_payload["slice_offset"] = float(scroll_result["slice_offset"])
                    if scroll_result.get("canonical_slice_offset") is not None:
                        view_payload["canonical_slice_offset"] = float(scroll_result["canonical_slice_offset"])
                    if scroll_result.get("canonical_axis_label") is not None:
                        view_payload["canonical_axis_label"] = scroll_result["canonical_axis_label"]
            except Exception as e:
                errors[view] = str(e)
            views[view] = view_payload

        payload: Dict[str, Any] = {"views": views, "errors": errors}
        self.session.update_state(initial_mid_slice_views=payload)
        return {"ok": not bool(errors), **payload}

    def reset_standard_views_to_initial_mid_slices(self, *, enabled: bool = True) -> Dict[str, Any]:
        """Return standard views to the initial middle slices before registration/fusion screenshots."""
        if not enabled or not self._reset_registration_preview_to_initial_slice_enabled():
            return {"ok": True, "skipped": True, "reason": "disabled"}

        stored = self.session.state.get("initial_mid_slice_views")
        views = stored.get("views") if isinstance(stored, dict) else None
        results: Dict[str, Any] = {"ok": True, "views": {}, "used_stored_offsets": bool(views)}
        for view in _STANDARD_VIEWS:
            payload = views.get(view) if isinstance(views, dict) else None
            orientation = str((payload or {}).get("orientation") or view)
            view_result: Dict[str, Any] = {"view": view, "orientation": orientation}
            try:
                view_result["orientation_result"] = self.set_slice_orientation(view=view, orientation=orientation)
                offset = (payload or {}).get("slice_offset")
                if offset is not None:
                    view_result["offset_result"] = self.set_slice_offset(view=view, offset=float(offset))
                else:
                    view_result["offset_result"] = self.set_slice_scroll_to(view=view, scroll_to=0.5)
            except Exception as e:
                results["ok"] = False
                view_result["error"] = str(e)
            results["views"][view] = view_result
        return results

    def bookmark_add(
        self,
        *,
        name: str,
        view: str = "axial",
        orientation: str = "axial",
        scroll_to: float = 0.5,
        size: int = 512,
    ) -> Dict[str, Any]:
        """Create a named bookmark: stores viewer_state + a representative slice PNG."""
        canonical_view = _canonical_view(view)
        args = {"name": name, "view": canonical_view, "orientation": orientation, "scroll_to": float(scroll_to), "size": int(size)}
        try:
            st = self.get_viewer_state()
            img = self.get_slice_png(view=canonical_view, orientation=orientation, scroll_to=scroll_to, size=size, out_name=f"bookmark_{name}.png")

            bookmarks = dict(self.session.state.get("bookmarks") or {})
            bookmarks[name] = {"viewer_state": st, "png_path": img.get("png_path")}
            self.session.update_state(bookmarks=bookmarks)

            result = {"name": name, "png_path": img.get("png_path")}
            self._log(tool="bookmark_add", args=args, result=result, artifacts={"png": img.get("png_path", "")})
            return result
        except Exception as e:
            self._log(tool="bookmark_add", args=args, result=None, ok=False, error=str(e))
            raise

    def bookmark_list(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            bookmarks = self.session.state.get("bookmarks") or {}
            result = {"names": sorted(list(bookmarks.keys()))}
            self._log(tool="bookmark_list", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="bookmark_list", args=args, result=None, ok=False, error=str(e))
            raise

    def _capture_current_view_png_raw(self, *, view: str = "axial", size: int = 512, out_name: Optional[str] = None) -> Dict[str, Any]:
        canonical_view = _canonical_view(view)
        st = self.get_viewer_state()
        v = canonical_view
        vs = (st.get("views") or {}).get(v) or {}
        orientation_raw = str(vs.get("orientation") or "axial")
        orientation = orientation_raw.strip().lower()
        if "ax" in orientation:
            orientation = "axial"
        elif "sag" in orientation:
            orientation = "sagittal"
        elif "cor" in orientation:
            orientation = "coronal"
        else:
            orientation = "axial"

        offset = vs.get("slice_offset")
        canonical_axis_label = vs.get("canonical_axis_label")
        canonical_offset = vs.get("canonical_slice_offset")
        if offset is None:
            if out_name is None:
                out_name = f"obs_raw_{v}_{orientation}_{int(size)}.png"
            out_path = self._artifact_path(out_name)
            self.client.save_slice_png(out_path, view=v, orientation=orientation, scroll_to=0.5, size=size)
            return {
                "viewer_state": st,
                "view": v,
                "orientation": orientation,
                "offset": None,
                "canonical_axis_label": canonical_axis_label,
                "canonical_offset": canonical_offset,
                "png_path": str(out_path),
            }

        if out_name is None:
            out_name = f"obs_raw_{v}_{orientation}_{float(offset):.3f}_{int(size)}.png"
        out_path = self._artifact_path(out_name)

        try:
            self.client.save_slice_png(
                out_path,
                view=v,
                orientation=orientation,
                scroll_to=0.5,
                size=size,
                offset=float(offset),
            )
        except SlicerRequestError as e:
            message = str(e)
            if "GetDataType" not in message:
                raise
            logger.warning("observe_view detected stale slice layers for view=%s; attempting recover_standard_views retry", v)
            active_volume_id = st.get("active_volume_id")
            active_volume_name = st.get("active_volume_name")
            self.recover_standard_views(volume_id=active_volume_id, volume_name=active_volume_name, centered=True, fit=True)
            st = self.get_viewer_state()
            vs = (st.get("views") or {}).get(v) or {}
            retry_offset = vs.get("slice_offset")
            if retry_offset is None:
                retry_offset = float(offset)
            self.client.save_slice_png(
                out_path,
                view=v,
                orientation=orientation,
                scroll_to=0.5,
                size=size,
                offset=float(retry_offset),
            )
            offset = retry_offset

        return {
            "viewer_state": st,
            "view": v,
            "orientation": orientation,
            "offset": (float(offset) if offset is not None else None),
            "canonical_axis_label": canonical_axis_label,
            "canonical_offset": (float(canonical_offset) if canonical_offset is not None else None),
            "png_path": str(out_path),
        }

    def observe_view(self, *, view: str = "axial", size: int = 512, out_name: Optional[str] = None) -> Dict[str, Any]:
        """Capture the *current* slice view as PNG.

        Unlike `get_slice_png(scroll_to=...)`, this tries to preserve the current interactive state
        (slice offset + current orientation) by querying `get_viewer_state()` and rendering at that offset.

        This is the building block for "auto" viewer mode: after actions like zoom/WL/scroll, you want
        an updated observation without forcing the agent to call `get_slice_png` manually.
        """
        canonical_view = _canonical_view(view)
        args = {"view": canonical_view, "size": int(size), "out_name": out_name}
        try:
            raw_capture = self._capture_current_view_png_raw(view=canonical_view, size=int(size), out_name=out_name)
            cropped_png_path, crop_box, full_size = self._crop_focus_png_in_place(png_path=str(raw_capture["png_path"]))
            cropped_png_path = self._stamp_slice_mm_on_png(
                png_path=str(cropped_png_path),
                view=canonical_view,
                offset=(float(raw_capture["offset"]) if raw_capture.get("offset") is not None else None),
                canonical_offset=(float(raw_capture["canonical_offset"]) if raw_capture.get("canonical_offset") is not None else None),
                axis_label=raw_capture.get("canonical_axis_label"),
            )
            self._remember_view_capture(
                view=raw_capture["view"],
                png_path=cropped_png_path,
                raw_png_path=str(raw_capture["png_path"]),
                crop_box=crop_box,
                full_size=full_size,
                orientation=str(raw_capture.get("orientation") or canonical_view),
                offset=(float(raw_capture["offset"]) if raw_capture.get("offset") is not None else None),
            )
            result = {
                "ok": True,
                "png_path": cropped_png_path,
                "view": raw_capture["view"],
                "orientation": raw_capture["orientation"],
                "offset": raw_capture["offset"],
                "canonical_axis_label": raw_capture.get("canonical_axis_label"),
                "canonical_offset": raw_capture.get("canonical_offset"),
                "crop_box": list(crop_box),
                "full_size": list(full_size),
            }
            self._log(tool="observe_view", args=args, result=result, artifacts={"png": cropped_png_path})
            return result
        except Exception as e:
            self._log(tool="observe_view", args=args, result=None, ok=False, error=str(e))
            raise

    def observe_standard_views(self, *, size: int = 512, out_prefix: str = "std") -> Dict[str, Any]:
        """Capture a standard 3-plane observation (axial/sagittal/coronal).

        Returns PNG paths for the current slice offsets.
        """
        args = {"size": int(size), "out_prefix": out_prefix}
        try:
            imgs: Dict[str, str] = {}
            for v in _STANDARD_VIEWS:
                res = self.observe_view(view=v, size=size, out_name=f"{out_prefix}_{v}.png")
                if isinstance(res, dict) and res.get("png_path"):
                    imgs[v] = res["png_path"]
            result = {"ok": True, "png_paths": imgs}
            self._log(tool="observe_standard_views", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="observe_standard_views", args=args, result=None, ok=False, error=str(e))
            raise

    def open_case(
        self,
        dicom_dir: Union[str, Path],
        *,
        preset: str = "mri_brain",
        layout: str = "four_up",
        linked_slices: bool = True,
        interpolation: bool = True,
        wl_auto: bool = True,
        baseline: bool = True,
        size: int = 512,
        clear_scene_first: bool = True,
        nifti_files: Optional[Sequence[Union[str, Path]]] = None,
        include_segmentations: bool = False,
        dicom_series_dirs: Optional[Sequence[Union[str, Path]]] = None,
        series_display_names: Optional[Sequence[str]] = None,
        active_prefer: Optional[Sequence[str]] = None,
        window_preset: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Open a case (DICOM folder/series or NIfTI file/folder) with a sensible default preset."""

        case_path = _as_path(dicom_dir)
        args = {
            "case_path": str(case_path),
            "preset": preset,
            "layout": layout,
            "linked_slices": bool(linked_slices),
            "interpolation": bool(interpolation),
            "wl_auto": bool(wl_auto),
            "baseline": bool(baseline),
            "size": int(size),
            "clear_scene_first": bool(clear_scene_first),
            "nifti_files": [str(_as_path(x)) for x in nifti_files] if nifti_files else None,
            "include_segmentations": bool(include_segmentations),
            "dicom_series_dirs": [str(_as_path(x)) for x in dicom_series_dirs] if dicom_series_dirs else None,
            "series_display_names": [str(x) for x in series_display_names] if series_display_names else None,
            "active_prefer": list(active_prefer) if active_prefer else None,
            "window_preset": window_preset,
        }

        summary: Dict[str, Any] = {"ok": True, "steps": {}, "baseline": None, "warnings": []}
        fatal_steps = {"load_dicom", "load_dicom_series", "load_nifti"}

        def _step(name: str, fn):
            try:
                r = fn()
                summary["steps"][name] = r
                if isinstance(r, dict) and r.get("ok") is False and name in fatal_steps:
                    summary["ok"] = False
                elif isinstance(r, dict) and r.get("ok") is False:
                    summary["warnings"].append({"step": name, "error": r.get("error") or r})
                return r
            except Exception as e:
                if name in fatal_steps:
                    summary["ok"] = False
                summary["steps"][name] = {"ok": False, "error": str(e)}
                summary["warnings"].append({"step": name, "error": str(e)})
                return None

        if clear_scene_first:
            _step("clear_scene", lambda: self.clear_scene())

        def _looks_like_nifti_dir(p: Path) -> bool:
            if not p.exists():
                return False
            if p.is_file():
                pl = str(p).lower()
                return pl.endswith(".nii") or pl.endswith(".nii.gz")
            if p.is_dir():
                for c in p.iterdir():
                    cl = str(c).lower()
                    if c.is_file() and (cl.endswith(".nii") or cl.endswith(".nii.gz")):
                        return True
            return False

        def _discover_nifti_files(dir_path: Path) -> List[Path]:
            all_files = sorted([p for p in dir_path.iterdir() if p.is_file() and str(p).lower().endswith((".nii", ".nii.gz"))])
            if not include_segmentations:
                all_files = [p for p in all_files if "segmentation" not in p.name.lower()]

            if preset != "mri_brain":
                return all_files

            def pick(modality: str, *, prefer_bias: bool = True) -> Optional[Path]:
                mod_l = modality.lower()
                cands = [p for p in all_files if mod_l in p.name.lower()]
                if modality.lower() == "t1":
                    cands = [p for p in cands if "t1c" not in p.name.lower()]
                if not cands:
                    return None
                if prefer_bias:
                    bias = [p for p in cands if "bias" in p.name.lower()]
                    if bias:
                        return sorted(bias)[0]
                return sorted(cands)[0]

            chosen: List[Path] = []
            for mod in ("t1c", "flair", "t2", "t1"):
                p = pick(mod)
                if p is not None:
                    chosen.append(p)
            return chosen if chosen else all_files

        load_res = None
        if dicom_series_dirs is not None:
            ds = [_as_path(x) for x in dicom_series_dirs]
            load_res = _step(
                "load_dicom_series",
                lambda: self.load_dicom_series(
                    [str(p) for p in ds],
                    clear_scene_first=False,
                    active_prefer=active_prefer,
                ),
            )
        elif nifti_files is not None:
            nf = [_as_path(x) for x in nifti_files]
            load_res = _step(
                "load_nifti",
                lambda: self.load_nifti(
                    [str(p) for p in nf],
                    clear_scene_first=False,
                    active_prefer=active_prefer or ["t1c", "flair", "t2", "t1"],
                ),
            )
        elif _looks_like_nifti_dir(case_path):
            if case_path.is_file():
                load_res = _step(
                    "load_nifti",
                    lambda: self.load_nifti(
                        str(case_path),
                        clear_scene_first=False,
                        active_prefer=active_prefer or ["t1c", "flair", "t2", "t1"],
                    ),
                )
            else:
                discovered = _discover_nifti_files(case_path)
                load_res = _step(
                    "load_nifti",
                    lambda: self.load_nifti(
                        [str(p) for p in discovered],
                        clear_scene_first=False,
                        active_prefer=active_prefer or ["t1c", "flair", "t2", "t1"],
                    ),
                )
        else:
            load_res = _step(
                "load_dicom",
                lambda: self.load_dicom(case_path, clear_scene_first=False, active_prefer=active_prefer),
            )

        _step("set_layout", lambda: self.set_layout(layout=layout))
        _step("set_linked_slices", lambda: self.set_linked_slices(enabled=linked_slices))
        _step("set_interpolation", lambda: self.set_interpolation(interpolate=interpolation))

        if window_preset:
            _step("apply_window_preset", lambda: self.apply_window_preset(preset=window_preset))
        elif wl_auto:
            _step("set_window_level_auto", lambda: self.set_window_level(auto=True))

        for v in _STANDARD_VIEWS:
            _step(f"fit_{v}", lambda v=v: self.fit_slice(view=v))

        _step("record_initial_mid_slice_views", self._record_initial_mid_slice_views)

        if isinstance(load_res, dict) and load_res.get("ok"):
            self.session.update_state(case_preset=preset)

        if baseline:
            try:
                b = {
                    "axial": self.get_slice_png(view="axial", orientation="axial", scroll_to=0.5, size=size, out_name="baseline_axial.png")["png_path"],
                    "sagittal": self.get_slice_png(view="sagittal", orientation="sagittal", scroll_to=0.5, size=size, out_name="baseline_sagittal.png")["png_path"],
                    "coronal": self.get_slice_png(view="coronal", orientation="coronal", scroll_to=0.5, size=size, out_name="baseline_coronal.png")["png_path"],
                }
                summary["baseline"] = b
            except Exception as e:
                summary["ok"] = False
                summary["baseline"] = {"ok": False, "error": str(e)}

        try:
            summary["viewer_state"] = self.get_viewer_state()
        except Exception as e:
            summary["ok"] = False
            summary["viewer_state"] = {"ok": False, "error": str(e)}

        try:
            scene_inventory = self.client.list_volumes()
            summary["scene_inventory"] = scene_inventory
            summary["scene_inventory_text"] = format_scene_inventory_for_prompt(
                scene_inventory,
                active_volume_id=self.session.state.get("active_volume_id"),
                active_volume_name=self.session.state.get("active_volume_name"),
            )
            self.session.update_state(
                scene_inventory=scene_inventory,
                scene_inventory_text=summary["scene_inventory_text"],
                loaded_series_dirs=args.get("dicom_series_dirs"),
                loaded_series_display_names=args.get("series_display_names"),
            )
        except Exception as e:
            summary["warnings"].append({"step": "scene_inventory", "error": str(e)})
            summary["scene_inventory"] = []
            summary["scene_inventory_text"] = f"Currently loaded scene volumes: unavailable ({e})"

        if args.get("dicom_series_dirs") is not None or args.get("series_display_names") is not None:
            self.session.update_state(
                loaded_series_dirs=args.get("dicom_series_dirs"),
                loaded_series_display_names=args.get("series_display_names"),
            )
        self._log(tool="open_case", args=args, result=summary)
        return summary

    def scroll_sweep(
        self,
        *,
        view: str = "axial",
        orientation: str = "axial",
        start: float = 0.0,
        end: float = 1.0,
        n_frames: int = 12,
        size: int = 512,
        output: str = "keyframes",
        fps: int = 12,
        analyze: bool = False,
        prompt: Optional[str] = None,
        auto_prompt: bool = True,
        out_prefix: str = "scroll",
    ) -> Dict[str, Any]:
        """Scroll/sweep through a stack and return observations.

        Keyframe contact sheets are annotated with the physical S/A/L position so the model can
        jump back to a specific slice later using set_slice_offset.
        """

        args = {
            "view": view,
            "orientation": orientation,
            "start": float(start),
            "end": float(end),
            "n_frames": int(n_frames),
            "size": int(size),
            "output": output,
            "fps": int(fps),
            "analyze": bool(analyze),
            "prompt": prompt,
            "auto_prompt": bool(auto_prompt),
            "out_prefix": out_prefix,
        }

        try:
            v = _canonical_view(view)
            ori = (orientation or "axial").lower()
            mode = (output or "keyframes").lower()

            if mode not in {"keyframes", "frames", "video"}:
                raise ValueError("output must be one of: keyframes, video")
            if n_frames < 2:
                raise ValueError("n_frames must be >= 2")

            axis_label = {"axial": "S", "sagittal": "R", "coronal": "A"}.get(v, "mm")
            offset_range = None
            try:
                offset_range = self._quiet_get_slice_offset_range(view=v)
                if isinstance(offset_range, dict) and offset_range.get("axis_label") is not None:
                    axis_label = str(offset_range.get("axis_label") or axis_label)
            except Exception:
                offset_range = None

            if mode in {"keyframes", "frames"}:
                frames_dir = self._artifact_path(f"{out_prefix}_{v}_{ori}_{n_frames}f")
                frames_dir.mkdir(parents=True, exist_ok=True)
                png_paths: List[str] = []
                ts: List[float] = []
                frame_offsets_mm: List[Optional[float]] = []
                lo = hi = None
                if isinstance(offset_range, dict):
                    lo = offset_range.get("min_offset")
                    hi = offset_range.get("max_offset")
                for i in range(n_frames):
                    t = float(start + (end - start) * (i / (n_frames - 1)))
                    ts.append(t)
                    out_path = frames_dir / f"frame_{i:05d}.png"
                    self.client.save_slice_png(out_path, view=v, orientation=ori, scroll_to=t, size=size)
                    if lo is not None and hi is not None:
                        off = float(lo) + t * (float(hi) - float(lo))
                        frame_offsets_mm.append(off)
                        self._stamp_slice_mm_on_png(png_path=str(out_path), view=v, canonical_offset=off, axis_label=axis_label)
                    else:
                        frame_offsets_mm.append(None)
                        self._stamp_slice_mm_on_png(png_path=str(out_path), view=v, scroll_to=t)
                    png_paths.append(str(out_path))

                contact_sheet_path: Optional[Path] = None
                try:
                    from PIL import Image  # type: ignore

                    imgs = [Image.open(p) for p in png_paths]
                    w, h = imgs[0].size
                    n_cols = 4
                    n_rows = (len(imgs) + n_cols - 1) // n_cols
                    sheet = Image.new("RGB", (w * n_cols, h * n_rows))
                    for idx, im in enumerate(imgs):
                        r = idx // n_cols
                        c = idx % n_cols
                        sheet.paste(im.convert("RGB"), (c * w, r * h))
                    contact_sheet_path = self._artifact_path(f"{out_prefix}_{v}_{ori}_{n_frames}f_contact.png")
                    sheet.save(contact_sheet_path)
                except Exception:
                    contact_sheet_path = None

                result = {
                    "ok": True,
                    "mode": "keyframes",
                    "frames_dir": str(frames_dir),
                    "png_paths": png_paths,
                    "t_values": ts,
                    "frame_offsets_mm": frame_offsets_mm,
                    "offset_axis": axis_label,
                    "offset_range": offset_range,
                    "contact_sheet_png_path": str(contact_sheet_path) if contact_sheet_path else None,
                }
                self._log(tool="scroll_sweep", args=args, result=result, artifacts={"frames_dir": str(frames_dir)})
                return result

            if self.video is None:
                raise RuntimeError("VideoRenderer is not configured (ffmpeg not available)")

            cine = self.capture_cine(
                view=v,
                orientation=ori,
                start=float(start),
                end=float(end),
                n_frames=int(n_frames),
                size=int(size),
                fps=int(fps),
                frames_dir_name=f"{out_prefix}_{v}_{ori}_{n_frames}f_frames",
                out_mp4_name=f"{out_prefix}_{v}_{ori}_{n_frames}f_{fps}fps.mp4",
            )
            result: Dict[str, Any] = {"ok": True, "mode": "video", **cine}

            if analyze:
                if self.gemini is None:
                    raise RuntimeError("Gemini analyzer is not configured (set GEMINI_API_KEY).")
                if prompt is None:
                    if not auto_prompt:
                        raise ValueError("prompt must be provided when analyze=True and auto_prompt=False")
                    prompt = render_prompt(RUNTIME_PROMPT_REF, "templates.scroll_sweep_analysis")
                text = self.gemini.analyze_mp4(video_path=result["mp4_path"], prompt=prompt)
                result["analysis_text"] = text
                result["analysis_model"] = getattr(self.gemini, "model", None)

            self._log(tool="scroll_sweep", args=args, result=result, artifacts={"mp4": str(result.get("mp4_path", ""))})
            return result
        except Exception as e:
            self._log(tool="scroll_sweep", args=args, result=None, ok=False, error=str(e))
            raise
