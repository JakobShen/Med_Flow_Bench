
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

class ToolContextQuantMixin:
    def roi_stats_ijk(
        self,
        *,
        ijk_min: Sequence[int],
        ijk_max: Sequence[int],
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Default to the currently selected (active) volume if not provided.
        if volume_id is None and volume_name is None:
            volume_id = self.session.state.get("active_volume_id")
            volume_name = self.session.state.get("active_volume_name")

        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "ijk_min": list(map(int, ijk_min)),
            "ijk_max": list(map(int, ijk_max)),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="roi_stats_ijk",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="roi_stats_ijk", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="roi_stats_ijk", args=args, result=None, ok=False, error=str(e))
            raise

    def roi_stats_ras_box(
        self,
        *,
        ras_min: Sequence[float],
        ras_max: Sequence[float],
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if volume_id is None and volume_name is None:
            volume_id = self.session.state.get("active_volume_id")
            volume_name = self.session.state.get("active_volume_name")
        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "ras_min": [float(v) for v in ras_min],
            "ras_max": [float(v) for v in ras_max],
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="roi_stats_ras_box",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="roi_stats_ras_box", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="roi_stats_ras_box", args=args, result=None, ok=False, error=str(e))
            raise

    def sample_intensity_ras(
        self,
        *,
        ras: Sequence[float],
        method: str = "nearest",
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if volume_id is None and volume_name is None:
            volume_id = self.session.state.get("active_volume_id")
            volume_name = self.session.state.get("active_volume_name")
        args = {
            "volume_id": volume_id,
            "volume_name": volume_name,
            "ras": [float(v) for v in ras],
            "method": str(method),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="sample_intensity_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="sample_intensity_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="sample_intensity_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def measure_distance_ras(self, *, p1: Sequence[float], p2: Sequence[float]) -> Dict[str, Any]:
        args = {"p1": [float(v) for v in p1], "p2": [float(v) for v in p2]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="measure_distance_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="measure_distance_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="measure_distance_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def measure_angle_ras(self, *, p1: Sequence[float], p2: Sequence[float], p3: Sequence[float]) -> Dict[str, Any]:
        args = {"p1": [float(v) for v in p1], "p2": [float(v) for v in p2], "p3": [float(v) for v in p3]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="measure_angle_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="measure_angle_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="measure_angle_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def measure_area_polygon_ras(self, *, points: Sequence[Sequence[float]]) -> Dict[str, Any]:
        args = {"points": [[float(v) for v in p] for p in points]}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="measure_area_polygon_ras",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="measure_area_polygon_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="measure_area_polygon_ras", args=args, result=None, ok=False, error=str(e))
            raise

    def set_thick_slab(self, *, view: str, enabled: bool, thickness_mm: float = 0.0, mode: str = "mip") -> Dict[str, Any]:
        args = {"view": view, "enabled": bool(enabled), "thickness_mm": float(thickness_mm), "mode": str(mode)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_thick_slab",
                args=args,
                session_id=self.session.session_id,
            )
            result = _normalize_public_view_payload(result)
            self._log(tool="set_thick_slab", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_thick_slab", args=args, result=None, ok=False, error=str(e))
            raise

    @staticmethod
    def _normalize_volume_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    def _volume_match_score(self, *, query: str, candidate_name: str) -> float:
        q_raw = str(query or "").strip()
        c_raw = str(candidate_name or "").strip()
        q_norm = self._normalize_volume_key(q_raw)
        c_norm = self._normalize_volume_key(c_raw)
        if not q_norm or not c_norm:
            return float("-inf")
        if q_raw == c_raw:
            return 10_000.0
        if q_norm == c_norm:
            return 9_000.0

        score = 0.0
        if q_norm in c_norm:
            score += 6_000.0 + min(len(q_norm), 200) / 10.0

        q_tokens = [tok for tok in q_norm.split() if tok]
        c_tokens = [tok for tok in c_norm.split() if tok]
        overlap = len(set(q_tokens).intersection(c_tokens))
        score += overlap * 300.0

        ratio = difflib.SequenceMatcher(None, q_norm, c_norm).ratio()
        score += ratio * 400.0

        q_is_pet = any(tok in q_norm for tok in ["pet", "suv", "uptake", "mac", "nac", "wb"])
        q_is_ct = any(tok in q_norm for tok in ["ct", "ctac", "thorax", "lung", "chest", "b70", "b40", "b45"])
        c_is_pet = any(tok in c_norm for tok in ["pet", "suv", "uptake", "body weight", "suvbw", "suvlbm", "suvbsa", "suvibw"])
        c_is_ct = any(tok in c_norm for tok in ["ct", "ctac", "thorax", "lung", "chest", "b70", "b40", "b45"])

        if q_is_pet:
            if c_is_pet:
                score += 700.0
            if "suvbw" in c_norm or "body weight" in c_norm:
                score += 450.0
            if "standardized uptake value body weight" in c_norm:
                score += 350.0
            if c_is_ct:
                score -= 600.0
        if q_is_ct:
            if c_is_ct:
                score += 700.0
            if "ctac" in c_norm:
                score += 450.0
            if c_is_pet:
                score -= 700.0

        if q_tokens and all(tok in c_norm for tok in q_tokens[:2]):
            score += 250.0
        return score

    def _resolve_volume_id(
        self,
        *,
        volume_ref: Optional[str] = None,
        volume_id: Optional[str] = None,
        volume_name: Optional[str] = None,
    ) -> str:
        if volume_id:
            return str(volume_id)

        vols = self.list_volumes()

        if volume_ref:
            ref = str(volume_ref)
            exact_id = [v for v in vols if str(v.get("id")) == ref]
            if exact_id:
                return str(exact_id[0]["id"])
            volume_name = ref

        if volume_name:
            exact = [v for v in vols if str(v.get("name")) == str(volume_name)]
            if exact:
                return str(exact[0]["id"])

            query = str(volume_name)
            ranked = sorted(
                (
                    (self._volume_match_score(query=query, candidate_name=str(v.get("name", ""))), v)
                    for v in vols
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if ranked and ranked[0][0] > 250.0:
                return str(ranked[0][1]["id"])
        raise ValueError(
            f"Could not resolve volume reference ref={volume_ref!r} id={volume_id!r} name={volume_name!r}. "
            "Use a loaded scene volume id or a volume name from the prompt-provided scene inventory."
        )

    def fuse_volumes(
        self,
        *,
        background_volume_ref: Optional[str] = None,
        background_volume_id: Optional[str] = None,
        background_volume_name: Optional[str] = None,
        foreground_volume_ref: Optional[str] = None,
        foreground_volume_id: Optional[str] = None,
        foreground_volume_name: Optional[str] = None,
        opacity: float = 0.35,
    ) -> Dict[str, Any]:
        bg_id = self._resolve_volume_id(
            volume_ref=background_volume_ref,
            volume_id=background_volume_id,
            volume_name=background_volume_name,
        )
        fg_id = self._resolve_volume_id(
            volume_ref=foreground_volume_ref,
            volume_id=foreground_volume_id,
            volume_name=foreground_volume_name,
        )
        return self.set_fusion(background_volume_id=bg_id, foreground_volume_id=fg_id, opacity=opacity)

    def set_fusion(self, *, background_volume_id: str, foreground_volume_id: str, opacity: float = 0.5) -> Dict[str, Any]:
        args = {"background_volume_id": background_volume_id, "foreground_volume_id": foreground_volume_id, "opacity": float(opacity)}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="set_fusion",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="set_fusion", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_fusion", args=args, result=None, ok=False, error=str(e))
            raise

    def clear_fusion(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="clear_fusion",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="clear_fusion", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="clear_fusion", args=args, result=None, ok=False, error=str(e))
            raise

    def resample_volume_isotropic(
        self,
        *,
        output_spacing_mm: float = 1.0,
        source_volume_id: Optional[str] = None,
        source_volume_name: Optional[str] = None,
        output_name: Optional[str] = None,
        interpolation_type: str = "linear",
        make_active: bool = True,
    ) -> Dict[str, Any]:
        args = {
            "output_spacing_mm": float(output_spacing_mm),
            "source_volume_id": source_volume_id or self.session.state.get("active_volume_id"),
            "source_volume_name": source_volume_name or self.session.state.get("active_volume_name"),
            "output_name": output_name,
            "interpolation_type": interpolation_type,
            "make_active": bool(make_active),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="resample_volume_isotropic",
                args=args,
                session_id=self.session.session_id,
                timeout_s=max(self.client.timeout_s, 180.0),
            )
            if isinstance(result, dict) and result.get("made_active"):
                self.session.update_state(
                    active_volume_id=result.get("output_volume_id"),
                    active_volume_name=result.get("output_volume_name"),
                )
            self._log(tool="resample_volume_isotropic", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="resample_volume_isotropic", args=args, result=None, ok=False, error=str(e))
            raise

    def compute_subtraction(self, *, volume_a_id: str, volume_b_id: str, output_name: str = "Subtraction") -> Dict[str, Any]:
        args = {"volume_a_id": volume_a_id, "volume_b_id": volume_b_id, "output_name": output_name}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="compute_subtraction",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="compute_subtraction", args=args, result=result)
            # Optionally update active volume in session (bridge sets it active)
            if result.get("ok"):
                out_id = result.get("output_volume_id")
                out_name = result.get("output_volume_name")
                if out_id:
                    self.session.update_state(active_volume_id=out_id, active_volume_name=out_name)
            return result
        except Exception as e:
            self._log(tool="compute_subtraction", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Export (bridge)
    # -------------------------

    def export_segmentation_dicom(
        self,
        *,
        segmentation_id: str,
        reference_volume_id: str,
        output_folder: Union[str, Path],
    ) -> Dict[str, Any]:
        output_folder_p = _as_path(output_folder)
        args = {
            "segmentation_id": segmentation_id,
            "reference_volume_id": reference_volume_id,
            "output_folder": str(output_folder_p),
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="export_segmentation_dicom",
                args=args,
                session_id=self.session.session_id,
            )
            self._log(tool="export_segmentation_dicom", args=args, result=result, artifacts={"folder": str(output_folder_p)})
            return result
        except Exception as e:
            self._log(tool="export_segmentation_dicom", args=args, result=None, ok=False, error=str(e))
            raise

