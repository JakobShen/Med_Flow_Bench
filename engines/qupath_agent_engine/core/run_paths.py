from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import local_timestamp_slug


@dataclass(slots=True)
class RunPaths:
    root: Path
    logs_dir: Path
    evidence_dir: Path
    midscene_dir: Path
    agent_dir: Path

    @classmethod
    def create(cls, *, base_dir: Path, run_prefix: str) -> "RunPaths":
        root = base_dir.expanduser().resolve() / f"{run_prefix}_{local_timestamp_slug()}"
        logs = root / "logs"
        evidence = root / "evidence"
        midscene = root / "midscene"
        agent = root / "agent"
        for path in (root, logs, evidence, midscene, agent):
            path.mkdir(parents=True, exist_ok=True)
        return cls(root=root, logs_dir=logs, evidence_dir=evidence, midscene_dir=midscene, agent_dir=agent)
