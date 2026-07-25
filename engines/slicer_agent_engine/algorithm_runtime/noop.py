from __future__ import annotations

from datetime import datetime, timezone

UTC = timezone.utc
from typing import Any, Dict
from uuid import uuid4

from .base import AlgorithmRuntime
from .models import AlgorithmJobRecord, AlgorithmRunResult


class NoopAlgorithmRuntime(AlgorithmRuntime):
    """Placeholder runtime used until a real expert runtime is wired in.

    The default behavior is intentionally explicit: callers get a structured
    failure instead of an attribute error or an implicit no-op.
    """

    backend_name = "noop"

    def run_blocking(self, *, operation: str, inputs: Dict[str, Any]) -> AlgorithmRunResult:
        job = self.submit(operation=operation, inputs=inputs)
        job.status = "failed"
        job.finished_at = datetime.now(UTC)
        job.error = f"No algorithm runtime configured for operation: {operation}"
        return AlgorithmRunResult(ok=False, job=job, error=job.error)

    def submit(self, *, operation: str, inputs: Dict[str, Any]) -> AlgorithmJobRecord:
        return AlgorithmJobRecord(
            job_id=f"noop-{uuid4()}",
            backend=self.backend_name,
            status="submitted",
            metadata={"operation": operation, "inputs": dict(inputs)},
        )

    def status(self, job_id: str) -> AlgorithmJobRecord:
        return AlgorithmJobRecord(job_id=job_id, backend=self.backend_name, status="unknown")

    def wait(self, job_id: str) -> AlgorithmRunResult:
        job = AlgorithmJobRecord(job_id=job_id, backend=self.backend_name, status="failed")
        job.finished_at = datetime.now(UTC)
        job.error = "No algorithm runtime configured"
        return AlgorithmRunResult(ok=False, job=job, error=job.error)

    def cancel(self, job_id: str) -> AlgorithmJobRecord:
        return AlgorithmJobRecord(job_id=job_id, backend=self.backend_name, status="cancelled")
