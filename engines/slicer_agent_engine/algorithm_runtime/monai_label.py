from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import requests

from .base import AlgorithmRuntime
from .models import AlgorithmArtifact, AlgorithmJobRecord, AlgorithmRunResult

UTC = timezone.utc


class MonaiLabelRuntime(AlgorithmRuntime):
    """Small blocking MONAI Label runtime used by agent toolpacks.

    The runtime intentionally stays independent of Slicer. It only talks to a
    MONAI Label HTTP server and returns files/artifacts for host-side tools to
    import back into Slicer.
    """

    backend_name = "monai_label"

    def __init__(
        self,
        *,
        server_url: Optional[str] = None,
        tmp_dir: Optional[str | Path] = None,
        timeout_s: float = 1800.0,
    ) -> None:
        self.server_url = str(server_url or os.environ.get("MONAI_LABEL_SERVER_URL") or "http://127.0.0.1:8001").rstrip("/")
        self.tmp_dir = Path(tmp_dir or os.environ.get("MONAI_LABEL_TMP_DIR") or "/tmp/slicer-agent-monai-label").expanduser()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self._jobs: Dict[str, AlgorithmJobRecord] = {}

    def _new_job(self, *, operation: str, inputs: Dict[str, Any]) -> AlgorithmJobRecord:
        job = AlgorithmJobRecord(
            job_id=f"monai-{uuid4().hex}",
            backend=self.backend_name,
            status="running",
            started_at=datetime.now(UTC),
            metadata={
                "operation": operation,
                "server_url": self.server_url,
                "inputs": {k: str(v) if isinstance(v, Path) else v for k, v in inputs.items()},
            },
        )
        self._jobs[job.job_id] = job
        return job

    def _finish(self, job: AlgorithmJobRecord, *, ok: bool, error: Optional[str] = None) -> None:
        job.status = "succeeded" if ok else "failed"
        job.finished_at = datetime.now(UTC)
        job.error = error

    def _info(self, job: AlgorithmJobRecord) -> AlgorithmRunResult:
        resp = requests.get(f"{self.server_url}/info/", timeout=min(self.timeout_s, 60.0))
        resp.raise_for_status()
        info = resp.json()
        self._finish(job, ok=True)
        return AlgorithmRunResult(
            ok=True,
            job=job,
            outputs={"server_url": self.server_url, "info": info},
        )

    def _infer_with_client(
        self,
        *,
        model: str,
        image_id: str,
        image_path: Path,
        params: Dict[str, Any],
        output_dir: Path,
    ) -> tuple[Optional[Path], Dict[str, Any]]:
        try:
            from monailabel.client.client import MONAILabelClient  # type: ignore
        except Exception:
            return self._infer_with_requests(
                model=model,
                image_id=image_id,
                image_path=image_path,
                params=params,
                output_dir=output_dir,
            )

        client = MONAILabelClient(self.server_url, tmpdir=str(output_dir))
        try:
            label_file, response_params = client.infer(model=model, image_id=image_id, params=params, file=str(image_path))
        except Exception:
            return self._infer_with_requests(
                model=model,
                image_id=image_id,
                image_path=image_path,
                params=params,
                output_dir=output_dir,
            )
        if not label_file and isinstance(response_params, (bytes, bytearray)):
            label_path = output_dir / f"{image_id}_{model}_label.nii.gz".replace("/", "-")
            label_path.write_bytes(bytes(response_params))
            return label_path, {"content_type": "application/octet-stream", "source": "monailabel_client_bytes"}
        label_path = Path(label_file).expanduser() if label_file else None
        if isinstance(response_params, dict):
            metadata = dict(response_params)
        elif isinstance(response_params, (bytes, bytearray)):
            metadata = {"content_type": "application/octet-stream", "bytes": len(response_params)}
        elif response_params:
            metadata = {"response_params": str(response_params)}
        else:
            metadata = {}
        return label_path, metadata

    def _infer_with_requests(
        self,
        *,
        model: str,
        image_id: str,
        image_path: Path,
        params: Dict[str, Any],
        output_dir: Path,
    ) -> tuple[Optional[Path], Dict[str, Any]]:
        # Fallback path for environments without monailabel.client installed.
        # MONAI Label normally returns multipart with a label file; this fallback
        # supports simple JSON responses and lets requests expose HTTP failures.
        url = f"{self.server_url}/infer/{model}"
        with image_path.open("rb") as f:
            resp = requests.post(
                url,
                params={"image": image_id},
                data={"params": json.dumps(params or {})},
                files={"file": (image_path.name, f, "application/octet-stream")},
                timeout=self.timeout_s,
            )
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            payload = resp.json()
            label_path = None
            if isinstance(payload, dict):
                label_path = payload.get("label_path") or payload.get("file") or payload.get("path")
            return (Path(label_path).expanduser() if label_path else None), payload
        out_path = output_dir / f"{image_id}_{model}_label.nii.gz".replace("/", "-")
        out_path.write_bytes(resp.content)
        return out_path, {"content_type": content_type}

    def _infer(self, job: AlgorithmJobRecord, inputs: Dict[str, Any]) -> AlgorithmRunResult:
        model = str(inputs.get("model") or os.environ.get("MONAI_LABEL_MODEL") or "vista3d")
        image_path = Path(str(inputs["image_path"])).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"MONAI Label image_path does not exist: {image_path}")

        image_id = str(inputs.get("image_id") or image_path.stem.replace(".nii", ""))
        params = dict(inputs.get("params") or {})
        output_dir = Path(inputs.get("output_dir") or self.tmp_dir / job.job_id).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        label_path, response_params = self._infer_with_client(
            model=model,
            image_id=image_id,
            image_path=image_path,
            params=params,
            output_dir=output_dir,
        )
        final_label_path: Optional[Path] = None
        if label_path is not None:
            label_path = label_path.expanduser().resolve()
            final_label_path = output_dir / f"{image_id}_{model}_label{''.join(label_path.suffixes) or '.nii.gz'}".replace("/", "-")
            if label_path != final_label_path:
                shutil.copy2(label_path, final_label_path)
            else:
                final_label_path = label_path

        self._finish(job, ok=True)
        artifacts = []
        if final_label_path is not None:
            artifacts.append(
                AlgorithmArtifact(
                    kind="labelmap",
                    path=str(final_label_path),
                    mime_type="application/gzip" if str(final_label_path).endswith(".gz") else "application/octet-stream",
                    description=f"MONAI Label output for {model}",
                )
            )
        return AlgorithmRunResult(
            ok=True,
            job=job,
            outputs={
                "model": model,
                "image_id": image_id,
                "image_path": str(image_path),
                "label_path": str(final_label_path) if final_label_path else None,
                "params": response_params,
            },
            artifacts=artifacts,
        )

    def run_blocking(self, *, operation: str, inputs: Dict[str, Any]) -> AlgorithmRunResult:
        job = self._new_job(operation=operation, inputs=inputs)
        try:
            op = str(operation).strip().lower()
            if op == "info":
                return self._info(job)
            if op in {"infer", "vista3d_segment"}:
                return self._infer(job, inputs)
            raise ValueError(f"Unsupported MONAI Label operation: {operation}")
        except Exception as e:
            self._finish(job, ok=False, error=str(e))
            return AlgorithmRunResult(ok=False, job=job, error=str(e))

    def submit(self, *, operation: str, inputs: Dict[str, Any]) -> AlgorithmJobRecord:
        job = AlgorithmJobRecord(
            job_id=f"monai-{uuid4().hex}",
            backend=self.backend_name,
            status="submitted",
            metadata={"operation": operation, "inputs": dict(inputs), "server_url": self.server_url},
        )
        self._jobs[job.job_id] = job
        return job

    def status(self, job_id: str) -> AlgorithmJobRecord:
        return self._jobs.get(job_id) or AlgorithmJobRecord(job_id=job_id, backend=self.backend_name, status="unknown")

    def wait(self, job_id: str) -> AlgorithmRunResult:
        job = self.status(job_id)
        return AlgorithmRunResult(ok=job.status == "succeeded", job=job, error=job.error)

    def cancel(self, job_id: str) -> AlgorithmJobRecord:
        job = self.status(job_id)
        job.status = "cancelled"
        job.finished_at = datetime.now(UTC)
        self._jobs[job.job_id] = job
        return job
