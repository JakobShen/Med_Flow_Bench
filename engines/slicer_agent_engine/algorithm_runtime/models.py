from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

UTC = timezone.utc
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProducedNodeRef:
    kind: str
    node_id: str
    name: Optional[str] = None
    role: Optional[str] = None


@dataclass(frozen=True)
class AlgorithmArtifact:
    kind: str
    path: str
    mime_type: Optional[str] = None
    description: Optional[str] = None


@dataclass
class AlgorithmJobRecord:
    job_id: str
    backend: str
    status: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgorithmRunResult:
    ok: bool
    job: AlgorithmJobRecord
    outputs: Dict[str, Any] = field(default_factory=dict)
    produced_nodes: List[ProducedNodeRef] = field(default_factory=list)
    artifacts: List[AlgorithmArtifact] = field(default_factory=list)
    error: Optional[str] = None
