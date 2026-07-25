
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

class ToolContextSceneMixin:
    def ping(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.get_system_version()
            self._log(tool="ping", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="ping", args=args, result=None, ok=False, error=str(e))
            raise

    def shutdown_slicer(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.shutdown_system()
            self._log(tool="shutdown_slicer", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="shutdown_slicer", args=args, result=None, ok=False, error=str(e))
            raise

    def clear_scene(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            # Prefer bridge (stable) if available, else fall back to client.clear_scene
            try:
                result = self.client.exec_bridge(bridge_dir=self.bridge_dir, tool="clear_scene", args={}, session_id=self.session.session_id)
            except Exception:
                result = self.client.clear_scene()
            self.session.update_state(active_volume_id=None, active_volume_name=None)
            self._log(tool="clear_scene", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="clear_scene", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Data loading / volume selection (bridge)
    # -------------------------

    def load_dicom(
        self,
        dicom_dir: Union[str, Path],
        *,
        clear_scene_first: bool = True,
        active_prefer: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        dicom_dir_p = _as_path(dicom_dir)
        args = {
            "dicom_dir": str(dicom_dir_p),
            "clear_scene_first": bool(clear_scene_first),
            "active_prefer": list(active_prefer) if active_prefer else None,
            "timeout_s": timeout_s,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="load_dicom",
                args={"dicom_dir": str(dicom_dir_p), "clear_scene_first": bool(clear_scene_first), "active_prefer": list(active_prefer) if active_prefer else None},
                session_id=self.session.session_id,
                timeout_s=timeout_s or self._estimate_load_timeout_s(n_series=1),
            )
            if result.get("ok"):
                self.session.update_state(
                    dicom_dir=str(dicom_dir_p),
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="load_dicom", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_dicom", args=args, result=None, ok=False, error=str(e))
            raise

    def load_dicom_series(
        self,
        dicom_dirs: Sequence[Union[str, Path]],
        *,
        clear_scene_first: bool = True,
        active_prefer: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        dicom_dirs_p = [str(_as_path(d)) for d in dicom_dirs]
        args = {
            "dicom_dirs": dicom_dirs_p,
            "clear_scene_first": bool(clear_scene_first),
            "active_prefer": list(active_prefer) if active_prefer else None,
            "timeout_s": timeout_s,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="load_dicom_series",
                args={"dicom_dirs": dicom_dirs_p, "clear_scene_first": bool(clear_scene_first), "active_prefer": list(active_prefer) if active_prefer else None},
                session_id=self.session.session_id,
                timeout_s=timeout_s or self._estimate_load_timeout_s(n_series=len(dicom_dirs_p)),
            )
            if result.get("ok"):
                self.session.update_state(
                    dicom_dir=dicom_dirs_p[0] if len(dicom_dirs_p) == 1 else None,
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="load_dicom_series", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_dicom_series", args=args, result=None, ok=False, error=str(e))
            raise

    def load_nifti(
        self,
        nifti: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        clear_scene_first: bool = True,
        active_prefer: Optional[Sequence[str]] = None,
        timeout_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Load NIfTI volume(s) from a file, a directory, or an explicit file list."""

        nifti_dir: Optional[str] = None
        nifti_files: Optional[List[str]] = None
        if isinstance(nifti, (str, Path)):
            p_in = _as_path(nifti)
            if p_in.is_dir():
                nifti_dir = str(p_in)
            else:
                nifti_files = [str(p_in)]
        else:
            nifti_files = [str(_as_path(x)) for x in nifti]

        args = {
            "nifti_dir": nifti_dir,
            "nifti_files": nifti_files,
            "clear_scene_first": bool(clear_scene_first),
            "active_prefer": list(active_prefer) if active_prefer else None,
            "timeout_s": timeout_s,
        }
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="load_nifti",
                args={"nifti_dir": nifti_dir, "nifti_files": nifti_files, "clear_scene_first": bool(clear_scene_first), "active_prefer": list(active_prefer) if active_prefer else None},
                session_id=self.session.session_id,
                timeout_s=timeout_s or self._estimate_load_timeout_s(n_series=len(nifti_files or ([nifti_dir] if nifti_dir else []))),
            )
            if result.get("ok"):
                self.session.update_state(
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="load_nifti", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_nifti", args=args, result=None, ok=False, error=str(e))
            raise

    def list_volumes(self) -> List[Dict[str, str]]:
        args: Dict[str, Any] = {}
        try:
            result = self.client.list_volumes()
            self._log(tool="list_volumes", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_volumes", args=args, result=None, ok=False, error=str(e))
            raise

    def select_volume(self, *, volume_id: Optional[str] = None, volume_name: Optional[str] = None) -> Dict[str, Any]:
        args = {"volume_id": volume_id, "volume_name": volume_name}
        try:
            result = self.client.exec_bridge(
                bridge_dir=self.bridge_dir,
                tool="select_volume",
                args=args,
                session_id=self.session.session_id,
            )
            if result.get("ok"):
                self.session.update_state(
                    active_volume_id=result.get("active_volume_id"),
                    active_volume_name=result.get("active_volume_name"),
                )
            self._log(tool="select_volume", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="select_volume", args=args, result=None, ok=False, error=str(e))
            raise
