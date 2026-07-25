from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from .models import AlgorithmJobRecord, AlgorithmRunResult


class AlgorithmRuntime(ABC):
    """Internal job-oriented runtime for long-running expert algorithms.

    This layer is intentionally not exposed directly to the model. Future expert
    integrations such as MONAI Label should plug into this boundary and then be
    wrapped by stable workflow tools.
    """

    @abstractmethod
    def run_blocking(self, *, operation: str, inputs: Dict[str, Any]) -> AlgorithmRunResult:
        raise NotImplementedError

    @abstractmethod
    def submit(self, *, operation: str, inputs: Dict[str, Any]) -> AlgorithmJobRecord:
        raise NotImplementedError

    @abstractmethod
    def status(self, job_id: str) -> AlgorithmJobRecord:
        raise NotImplementedError

    @abstractmethod
    def wait(self, job_id: str) -> AlgorithmRunResult:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job_id: str) -> AlgorithmJobRecord:
        raise NotImplementedError
