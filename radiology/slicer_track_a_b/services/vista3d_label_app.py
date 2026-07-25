#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small MONAI-Label-like VISTA3D HTTP wrapper.")
    parser.add_argument("--bundle_dir", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "image"))


def _coerce_label_prompt(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, (int, float, str)):
        value = [value]
    return [int(x) for x in value]


def _coerce_points(value: Any) -> list[list[float]]:
    if value is None or value == "":
        return []
    points: list[list[float]] = []
    for point in value:
        vals = [float(x) for x in point]
        if len(vals) != 3:
            raise ValueError("Each point must be length-3 original image-space IJK [x,y,z]")
        points.append(vals)
    return points


def create_app(*, bundle_dir: Path, work_dir: Path) -> FastAPI:
    app = FastAPI(title="VISTA3D Label App")
    bundle_dir = bundle_dir.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/info/")
    def info() -> Dict[str, Any]:
        return {
            "name": "vista3d_label_app",
            "server": "monai-label-compatible",
            "models": {
                "vista3d": {
                    "name": "vista3d",
                    "type": "segmentation",
                    "bundle_dir": str(bundle_dir),
                    "description": "MONAI VISTA3D bundle inference wrapper.",
                }
            },
        }

    @app.post("/infer/{model}")
    async def infer(model: str, image: str = "image", params: str = Form("{}"), file: UploadFile = File(...)):
        if model != "vista3d":
            raise HTTPException(status_code=404, detail=f"Unsupported model: {model}")
        if not (bundle_dir / "configs" / "inference.json").exists():
            raise HTTPException(status_code=500, detail=f"Missing VISTA3D inference config under {bundle_dir}")

        try:
            parsed_params = json.loads(params or "{}")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid params JSON: {exc}") from exc
        try:
            label_prompt = _coerce_label_prompt(parsed_params.get("label_prompt"))
            points = _coerce_points(parsed_params.get("points"))
            point_labels = [int(x) for x in (parsed_params.get("point_labels") or [])]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid VISTA3D prompt params: {exc}") from exc
        if points and len(point_labels) != len(points):
            raise HTTPException(status_code=400, detail="point_labels must have the same length as points")
        if points and label_prompt and len(label_prompt) != 1:
            raise HTTPException(status_code=400, detail="VISTA3D point prompts may be combined with at most one label_prompt")
        run_id = f"{int(time.time())}_{_safe_name(image)}"
        run_dir = work_dir / run_id
        input_dir = run_dir / "input"
        output_dir = run_dir / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".nii.gz" if str(file.filename or "").endswith(".nii.gz") else ".nii"
        image_path = input_dir / f"{_safe_name(image)}{suffix}"
        with image_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        input_dict: Dict[str, Any] = {"image": str(image_path)}
        if label_prompt:
            input_dict["label_prompt"] = label_prompt
        if points:
            input_dict["points"] = points
            input_dict["point_labels"] = point_labels
        cmd = [
            sys.executable,
            "-m",
            "monai.bundle",
            "run",
            "--config_file",
            "configs/inference.json",
            "--input_dict",
            repr(input_dict),
            "--output_dir",
            str(output_dir),
        ]
        proc = subprocess.run(cmd, cwd=str(bundle_dir), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run_dir / "vista3d_infer.log").write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"VISTA3D inference failed; see {run_dir / 'vista3d_infer.log'}")

        candidates = sorted(
            [p for p in output_dir.rglob("*") if p.is_file() and (str(p).endswith(".nii") or str(p).endswith(".nii.gz"))],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise HTTPException(status_code=500, detail=f"VISTA3D produced no NIfTI output under {output_dir}")
        label_path = candidates[0]
        return FileResponse(str(label_path), media_type="application/octet-stream", filename=label_path.name)

    return app


def main() -> None:
    args = parse_args()
    app = create_app(bundle_dir=Path(args.bundle_dir), work_dir=Path(args.work_dir))
    uvicorn.run(app, host=args.host, port=int(args.port), access_log=False)


if __name__ == "__main__":
    main()
