from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from ..algorithm_runtime import MonaiLabelRuntime
from ..monai_label_tools import DEFAULT_VISTA3D_LABEL_OPACITY, MonaiLabelToolService


class ToolContextMonaiLabelMixin:
    def _monai_label_runtime(self) -> MonaiLabelRuntime:
        runtime = getattr(self, "algorithm_runtime", None)
        if isinstance(runtime, MonaiLabelRuntime):
            return runtime
        runtime = MonaiLabelRuntime(
            server_url=os.environ.get("MONAI_LABEL_SERVER_URL") or "http://127.0.0.1:8001",
            tmp_dir=self.session.out_dir / "monai_label",
        )
        self.algorithm_runtime = runtime
        return runtime

    def _monai_label_tools(self) -> MonaiLabelToolService:
        return MonaiLabelToolService(ctx=self, runtime_factory=self._monai_label_runtime)

    def monai_label_status(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self._monai_label_tools().status()
            self._log(tool="monai_label_status", args=args, result=result.payload, ok=result.ok, error=result.error)
            return result.payload
        except Exception as e:
            self._log(tool="monai_label_status", args=args, result=None, ok=False, error=str(e))
            raise

    def vista3d_label_catalog(self, *, query: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        args = {"query": query, "limit": int(limit)}
        try:
            result = self._monai_label_tools().label_catalog(query=query, limit=int(limit))
            self._log(tool="vista3d_label_catalog", args=args, result=result.payload, ok=result.ok, error=result.error)
            return result.payload
        except Exception as e:
            self._log(tool="vista3d_label_catalog", args=args, result=None, ok=False, error=str(e))
            raise

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
    ) -> Dict[str, Any]:
        args = {
            "source_volume_id": source_volume_id,
            "source_volume_name": source_volume_name,
            "model": model,
            "label_prompt": list(label_prompt) if label_prompt is not None else None,
            "points_ijk": points_ijk,
            "points_ras": points_ras,
            "point_labels": point_labels,
            "seed_view": seed_view,
            "seed_bbox_1000": seed_bbox_1000,
            "seed_point_1000": seed_point_1000,
            "point_label": int(point_label),
            "segmentation_name": segmentation_name,
            "size": int(size),
            "auto_show": bool(auto_show),
            "auto_center": bool(auto_center),
            "opacity": float(opacity),
            "screen_views": list(screen_views) if screen_views is not None else ["axial", "sagittal", "coronal"],
        }
        try:
            result = self._monai_label_tools().vista3d_segment(
                source_volume_id=source_volume_id,
                source_volume_name=source_volume_name,
                model=model,
                label_prompt=label_prompt,
                points_ijk=points_ijk,
                points_ras=points_ras,
                point_labels=point_labels,
                seed_view=seed_view,
                seed_bbox_1000=seed_bbox_1000,
                seed_point_1000=seed_point_1000,
                point_label=int(point_label),
                segmentation_name=segmentation_name,
                size=int(size),
                auto_show=bool(auto_show),
                auto_center=bool(auto_center),
                opacity=float(opacity),
                screen_views=screen_views,
            )
            self._log(tool="vista3d_segment", args=args, result=result.payload, ok=result.ok, error=result.error, artifacts=result.artifacts)
            return result.payload
        except Exception as e:
            self._log(tool="vista3d_segment", args=args, result=None, ok=False, error=str(e))
            raise

    def capture_segmentation_overlay(
        self,
        *,
        view: str = "axial",
        segmentation_id: Optional[str] = None,
        segmentation_name: Optional[str] = None,
        segment_id: Optional[str] = None,
        segment_name: Optional[str] = None,
        out_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "view": view,
            "segmentation_id": segmentation_id or self.session.state.get("active_segmentation_id"),
            "segmentation_name": segmentation_name or self.session.state.get("active_segmentation_name"),
            "segment_id": segment_id or self.session.state.get("active_segment_id"),
            "segment_name": segment_name or self.session.state.get("active_segment_name"),
            "out_name": out_name,
        }
        try:
            result = self._monai_label_tools().capture_segmentation_overlay(
                view=view,
                segmentation_id=args["segmentation_id"],
                segmentation_name=args["segmentation_name"],
                segment_id=args["segment_id"],
                segment_name=args["segment_name"],
                out_name=out_name,
            )
            self._log(tool="capture_segmentation_overlay", args=args, result=result.payload, ok=result.ok, error=result.error, artifacts=result.artifacts)
            return result.payload
        except Exception as e:
            self._log(tool="capture_segmentation_overlay", args=args, result=None, ok=False, error=str(e))
            raise
