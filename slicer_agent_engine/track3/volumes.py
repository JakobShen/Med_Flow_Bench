from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .helpers import safe_stem, uniform_indices


_AXIS_LABEL_BY_PLANE = {
    "axial": "S",
    "sagittal": "R",
    "coronal": "A",
}


@dataclass(frozen=True)
class LoadedVolume:
    path: Path
    name: str
    stem: str
    data_ijk: np.ndarray
    affine_ijk_to_ras: np.ndarray
    display_uint8: np.ndarray

    @property
    def shape_ijk(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.data_ijk.shape)

    def depth_for_plane(self, plane: str) -> int:
        plane_l = str(plane).strip().lower()
        if plane_l == "axial":
            return int(self.data_ijk.shape[2])
        if plane_l == "sagittal":
            return int(self.data_ijk.shape[0])
        if plane_l == "coronal":
            return int(self.data_ijk.shape[1])
        raise ValueError(f"Unsupported plane: {plane!r}")

    def slice_axis_label_and_mm(self, plane: str, index: int) -> Tuple[str, float]:
        plane_l = str(plane).strip().lower()
        shape = self.shape_ijk
        ci = (shape[0] - 1) / 2.0
        cj = (shape[1] - 1) / 2.0
        ck = (shape[2] - 1) / 2.0
        if plane_l == "axial":
            point = np.asarray([ci, cj, float(index), 1.0], dtype=float)
            ras = self.affine_ijk_to_ras @ point
            return "S", float(ras[2])
        if plane_l == "sagittal":
            point = np.asarray([float(index), cj, ck, 1.0], dtype=float)
            ras = self.affine_ijk_to_ras @ point
            return "R", float(ras[0])
        if plane_l == "coronal":
            point = np.asarray([ci, float(index), ck, 1.0], dtype=float)
            ras = self.affine_ijk_to_ras @ point
            return "A", float(ras[1])
        raise ValueError(f"Unsupported plane: {plane!r}")

    def slice_2d_uint8(self, plane: str, index: int) -> np.ndarray:
        plane_l = str(plane).strip().lower()
        idx = int(index)
        if plane_l == "axial":
            arr = self.display_uint8[:, :, idx]
        elif plane_l == "sagittal":
            arr = self.display_uint8[idx, :, :]
        elif plane_l == "coronal":
            arr = self.display_uint8[:, idx, :]
        else:
            raise ValueError(f"Unsupported plane: {plane!r}")
        out = np.asarray(arr, dtype=np.uint8)
        out = np.flipud(out.T)
        return out

    def sample_indices(self, plane: str, count: int) -> List[int]:
        return uniform_indices(self.depth_for_plane(plane), count)


def _normalize_uint8(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if not np.isfinite(lo):
        lo = float(finite.min())
    if not np.isfinite(hi):
        hi = float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(arr, lo, hi)
    scaled = (clipped - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255.0).round().astype(np.uint8)


def load_nifti_volume(path: str | Path) -> LoadedVolume:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "Track3 volume loading requires nibabel. Install it in the Track3 environment, e.g. `pip install nibabel`."
        ) from e

    path_p = Path(path).expanduser().resolve()
    img = nib.load(str(path_p))
    data = np.asarray(img.get_fdata(dtype=np.float32))
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D volume at {path_p}, got shape {list(data.shape)}")
    affine = np.asarray(img.affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"Unexpected affine shape for {path_p}: {affine.shape}")
    return LoadedVolume(
        path=path_p,
        name=path_p.name,
        stem=safe_stem(path_p),
        data_ijk=data,
        affine_ijk_to_ras=affine,
        display_uint8=_normalize_uint8(data),
    )


def load_case_volumes(paths: Sequence[str | Path]) -> List[LoadedVolume]:
    return [load_nifti_volume(path) for path in paths]


def plane_axis_label(plane: str) -> str:
    plane_l = str(plane).strip().lower()
    if plane_l not in _AXIS_LABEL_BY_PLANE:
        raise ValueError(f"Unsupported plane: {plane!r}")
    return _AXIS_LABEL_BY_PLANE[plane_l]


def build_volume_summary(volumes: Sequence[LoadedVolume], *, planes: Sequence[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for volume in volumes:
        plane_depths = {str(plane): int(volume.depth_for_plane(plane)) for plane in planes}
        out.append(
            {
                "path": str(volume.path),
                "name": volume.name,
                "stem": volume.stem,
                "shape_ijk": [int(v) for v in volume.shape_ijk],
                "planes": plane_depths,
            }
        )
    return out


__all__ = [
    "LoadedVolume",
    "build_volume_summary",
    "load_case_volumes",
    "load_nifti_volume",
    "plane_axis_label",
]
