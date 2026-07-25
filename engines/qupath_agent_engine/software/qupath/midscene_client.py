from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qupath_agent_engine.core.utils import compact

from .config import QuPathConfig


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(slots=True)
class MidsceneCallResult:
    ok: bool
    payload: dict[str, Any]


class MidsceneBridge:
    """Small CLI bridge to an isolated Node sidecar.

    The Python host deliberately speaks only a tiny contract, so the sidecar can be
    replaced later without rewriting the rest of the repo.
    """

    def __init__(self, config: QuPathConfig) -> None:
        self.config = config.normalized()

    def env_check(self) -> MidsceneCallResult:
        return self._invoke(command="env-check", payload={})

    def act(self, *, instruction: str, report_path: Path, step_log_file: Path | None = None) -> MidsceneCallResult:
        return self._invoke(
            command="act",
            payload={
                "instruction": instruction,
                "reportFileName": str(report_path),
                "stepLogFile": str(step_log_file) if step_log_file else None,
            },
        )

    def query(self, *, question: str, report_path: Path, step_log_file: Path | None = None) -> MidsceneCallResult:
        return self._invoke(
            command="query",
            payload={
                "question": question,
                "reportFileName": str(report_path),
                "stepLogFile": str(step_log_file) if step_log_file else None,
            },
        )

    def assert_condition(self, *, assertion: str, report_path: Path, step_log_file: Path | None = None) -> MidsceneCallResult:
        return self._invoke(
            command="assert",
            payload={
                "assertion": assertion,
                "reportFileName": str(report_path),
                "stepLogFile": str(step_log_file) if step_log_file else None,
            },
        )

    def _invoke(self, *, command: str, payload: dict[str, Any]) -> MidsceneCallResult:
        if self.config.sidecar_dir is None:
            raise RuntimeError(
                "Midscene sidecar directory is not configured. Set MIDSCENE_QUPATH_SIDECAR_DIR or pass --sidecar-dir."
            )

        sidecar_dir = self.config.sidecar_dir
        cli_path = sidecar_dir / "src" / "cli.ts"
        if not cli_path.exists():
            raise FileNotFoundError(f"Midscene sidecar CLI not found: {cli_path}")

        if shutil.which("npx") is None:
            raise RuntimeError("npx is required to run the Midscene sidecar.")

        merged_payload = {
            **payload,
            "qupathExecutable": str(self.config.executable_path),
            "qupathProcessName": self.config.process_name,
            "displayId": self.config.display_id,
            "displayDpr": self.config.display_dpr,
            "waitAfterActionMs": self.config.midscene_wait_after_action_ms,
            "replanningCycleLimit": self.config.midscene_replanning_cycle_limit,
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            json.dump(merged_payload, fh, ensure_ascii=False)
            payload_path = Path(fh.name)

        result_file_fd, result_file_name = tempfile.mkstemp(suffix=".json")
        os.close(result_file_fd)
        result_file_path = Path(result_file_name)

        env = os.environ.copy()
        report_name = merged_payload.get("reportFileName")
        if isinstance(report_name, str) and report_name.strip():
            report_path = Path(report_name).expanduser()
            env["MIDSCENE_RUN_DIR"] = str(report_path.parent)
            merged_payload["reportFileName"] = report_path.stem

        merged_payload["resultFile"] = str(result_file_path)
        payload_path.write_text(json.dumps(merged_payload, ensure_ascii=False), encoding="utf-8")

        cmd = [
            "npx",
            "--no-install",
            "tsx",
            str(cli_path),
            command,
            "--payload-file",
            str(payload_path),
        ]
        completed = subprocess.run(
            cmd,
            cwd=sidecar_dir,
            env=env,
            capture_output=True,
            text=True,
        )
        try:
            payload_path.unlink(missing_ok=True)
        except Exception:
            pass

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        result_data = self._read_result_file(result_file_path)
        try:
            result_file_path.unlink(missing_ok=True)
        except Exception:
            pass

        if completed.returncode != 0:
            base_payload = result_data or {}
            return MidsceneCallResult(
                ok=False,
                payload={
                    **base_payload,
                    "ok": False,
                    "error": (
                        base_payload.get("error")
                        or f"Midscene sidecar failed ({completed.returncode}): {stderr.strip() or stdout.strip()}"
                    ),
                    "command": command,
                    "stdout": compact(stdout),
                    "stderr": compact(stderr),
                    "text_output": self._text_output(command=command, data=base_payload, stdout=stdout, stderr=stderr),
                },
            )

        data = result_data
        if data is None:
            try:
                data = self._parse_stdout_json(stdout)
            except json.JSONDecodeError as exc:
                return MidsceneCallResult(
                    ok=False,
                    payload={
                        "ok": False,
                        "error": f"Failed to parse Midscene output: {exc}",
                        "command": command,
                        "stdout": compact(stdout),
                        "stderr": compact(stderr),
                        "text_output": self._text_output(command=command, data=None, stdout=stdout, stderr=stderr),
                    },
                )

        enriched = {
            **data,
            "command": command,
            "stdout": compact(stdout),
            "stderr": compact(stderr),
            "text_output": self._text_output(command=command, data=data, stdout=stdout, stderr=stderr),
        }
        return MidsceneCallResult(ok=bool(enriched.get("ok", True)), payload=enriched)

    @staticmethod
    def _read_result_file(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    @staticmethod
    def _parse_stdout_json(stdout: str) -> dict[str, Any]:
        text = ANSI_RE.sub("", stdout or "").strip()
        if not text:
            raise json.JSONDecodeError("empty stdout", stdout, 0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed

        start = text.rfind("{")
        while start != -1:
            candidate = text[start:]
            try:
                parsed = json.loads(candidate)
            except Exception:
                start = text.rfind("{", 0, start)
                continue
            if isinstance(parsed, dict):
                return parsed
            start = text.rfind("{", 0, start)

        raise json.JSONDecodeError("No JSON object found in stdout", stdout, 0)

    @staticmethod
    def _text_output(command: str, data: dict[str, Any] | None, stdout: str, stderr: str) -> str:
        if isinstance(data, dict):
            if isinstance(data.get("text_output"), str) and data["text_output"].strip():
                return data["text_output"].strip()
            if isinstance(data.get("result"), str) and data["result"].strip():
                return data["result"].strip()
            if command == "assert" and data.get("ok"):
                return "Assertion succeeded."
            error = data.get("error")
            if isinstance(error, str) and error.strip():
                return error.strip()
            result = data.get("result")
            if result is not None:
                return compact(result, max_chars=2000) if isinstance(result, str) else json.dumps(compact(result, max_chars=2000), ensure_ascii=False)
        cleaned_stdout = ANSI_RE.sub("", stdout or "").strip()
        if cleaned_stdout:
            return cleaned_stdout.splitlines()[-1].strip()
        cleaned_stderr = ANSI_RE.sub("", stderr or "").strip()
        return cleaned_stderr or ""
