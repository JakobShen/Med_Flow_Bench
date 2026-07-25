from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import shutil

from .run_logger import RunLogger
from .run_paths import RunPaths
from .utils import dump_json


@dataclass(slots=True)
class RunContext:
    run_paths: RunPaths
    logger: RunLogger
    benchmark_logger: RunLogger | None = None
    save_evidence: bool = False
    step_index: int = 0
    capture_index: int = 0
    summary: dict[str, Any] = field(default_factory=dict)

    def next_step_id(self) -> str:
        self.step_index += 1
        return f"step_{self.step_index:04d}"

    def next_capture_id(self) -> int:
        self.capture_index += 1
        return self.capture_index

    def evidence_path(self, name: str) -> Path:
        return self.run_paths.evidence_dir / name

    def midscene_path(self, name: str) -> Path:
        return self.run_paths.midscene_dir / name

    def agent_path(self, name: str) -> Path:
        return self.run_paths.agent_dir / name

    def finalize(self) -> None:
        if not self.save_evidence and self.run_paths.evidence_dir.exists():
            shutil.rmtree(self.run_paths.evidence_dir, ignore_errors=True)
            self.summary["evidence_persisted"] = False
        else:
            self.summary["evidence_persisted"] = True
        dump_json(self.run_paths.root / "summary.json", self.summary)
