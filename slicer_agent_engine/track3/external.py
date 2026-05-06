from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from slicer_agent_engine.benchmarking.judge import extract_choice_from_set

from .helpers import ensure_dir, majority_vote


class ExternalWorkerError(RuntimeError):
    pass


def run_external_worker(
    *,
    worker_script: str | Path,
    request_payload: Dict[str, Any],
    out_dir: str | Path,
    python_executable: Optional[str] = None,
    timeout_sec: float = 1800.0,
    session: Optional[Any] = None,
    event_prefix: str = "track3_worker",
) -> Dict[str, Any]:
    out_dir_p = ensure_dir(out_dir)
    worker_path = Path(worker_script).expanduser().resolve()
    if not worker_path.exists():
        raise ExternalWorkerError(f"Worker script does not exist: {worker_path}")

    req_path = out_dir_p / "request.json"
    resp_path = out_dir_p / "response.json"
    stdout_path = out_dir_p / "stdout.txt"
    stderr_path = out_dir_p / "stderr.txt"
    req_path.write_text(json.dumps(request_payload, indent=2), encoding="utf-8")

    py_exec = str(python_executable or sys.executable)
    cmd = [py_exec, str(worker_path), "--input_json", str(req_path), "--output_json", str(resp_path)]

    if session is not None and hasattr(session, "log_run_event"):
        session.log_run_event(
            f"{event_prefix}_start",
            worker_script=str(worker_path),
            python_executable=py_exec,
            request_json=str(req_path),
            response_json=str(resp_path),
            timeout_sec=float(timeout_sec),
            cmd=cmd,
        )

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            timeout=float(timeout_sec),
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise ExternalWorkerError(
            f"External worker timed out after {timeout_sec:.1f}s: {worker_path}"
        ) from e

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")

    if proc.returncode != 0:
        raise ExternalWorkerError(
            f"External worker failed with return code {proc.returncode}: {worker_path}\n"
            f"See {stderr_path} for stderr and {stdout_path} for stdout."
        )
    if not resp_path.exists():
        raise ExternalWorkerError(
            f"External worker did not produce {resp_path}. See {stdout_path} and {stderr_path}."
        )

    try:
        payload = json.loads(resp_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ExternalWorkerError(f"Failed to parse worker response JSON: {resp_path}") from e

    if session is not None and hasattr(session, "log_run_event"):
        session.log_run_event(
            f"{event_prefix}_done",
            worker_script=str(worker_path),
            python_executable=py_exec,
            return_code=int(proc.returncode),
            response_json=str(resp_path),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            response_preview=str(payload)[:1200],
        )

    return payload


def aggregate_sequence_outputs(
    outputs: Sequence[Dict[str, Any]],
    *,
    valid_choices: Sequence[str] = ("A", "B", "C"),
) -> Dict[str, Any]:
    labeled_choices: List[tuple[str, Optional[str]]] = []
    normalized_outputs: List[Dict[str, Any]] = []
    for idx, raw in enumerate(outputs, start=1):
        text = str(raw.get("text") or "")
        label = str(raw.get("label") or raw.get("sequence_label") or raw.get("volume_name") or f"run_{idx}")
        choice = extract_choice_from_set(text, valid_choices=valid_choices)
        enriched = dict(raw)
        enriched["parsed_choice"] = choice
        normalized_outputs.append(enriched)
        labeled_choices.append((label, choice))

    final_choice, counts = majority_vote(labeled_choices, valid_choices=valid_choices)
    winning_output: Optional[Dict[str, Any]] = None
    if final_choice is not None:
        for item in normalized_outputs:
            if item.get("parsed_choice") == final_choice:
                winning_output = item
                break
    if winning_output is None and normalized_outputs:
        winning_output = normalized_outputs[0]

    base_text = str((winning_output or {}).get("text") or "")
    vote_lines = [
        "SEQUENCE_VOTE_SUMMARY:",
        *[
            f"- {str(item.get('label') or item.get('sequence_label') or item.get('volume_name') or f'run_{i+1}')}: {str(item.get('parsed_choice') or '<none>')}"
            for i, item in enumerate(normalized_outputs)
        ],
        f"- FINAL_VOTE: {final_choice or '<none>'}",
        f"- VOTE_COUNTS: {counts}",
    ]

    final_text = base_text.strip()
    if final_text:
        final_text += "\n\n"
    final_text += "\n".join(vote_lines)
    if final_choice is not None:
        final_text += f"\nANSWER: {final_choice}"

    return {
        "choice": final_choice,
        "counts": counts,
        "outputs": normalized_outputs,
        "text": final_text,
        "winning_output": winning_output,
    }


__all__ = [
    "ExternalWorkerError",
    "aggregate_sequence_outputs",
    "run_external_worker",
]
