from __future__ import annotations

import json
import platform
import shutil
import subprocess
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlencode, urlparse

from qupath_agent_engine.core.types import CaseSpec
from qupath_agent_engine.core.utils import sleep_seconds

from .config import QuPathConfig
from .platform import macos, windows


class QuPathLauncher:
    def __init__(self, config: QuPathConfig) -> None:
        self.config = config.normalized()
        self._platform = platform.system().lower()
        self._target_pid: int | None = None

    @property
    def process_name(self) -> str:
        return self.config.process_name or self.config.executable_path.name

    def launch_app_only(self) -> dict[str, Any]:
        if self._platform == "darwin":
            app_bundle = macos.derive_app_bundle(self.config.executable_path)
            macos.open_with_app(app_bundle)
            sleep_seconds(self.config.load_wait_seconds)
            try:
                macos.activate_process(self.process_name)
            except Exception:
                pass
            return {
                "ok": True,
                "platform": self._platform,
                "app_bundle": str(app_bundle),
                "process_name": self.process_name,
            }

        if self._platform == "windows":
            windows.open_with_executable(self.config.executable_path)
            sleep_seconds(self.config.load_wait_seconds)
            windows.activate_process(self.process_name)
            return {
                "ok": True,
                "platform": self._platform,
                "process_name": self.process_name,
            }

        raise NotImplementedError(f"Unsupported platform: {self._platform}")

    def focus(self) -> None:
        if self._platform == "darwin":
            try:
                macos.activate_process(self.process_name)
            except Exception:
                pass
            return
        if self._platform == "windows":
            try:
                windows.activate_process(self.process_name)
            except Exception:
                pass
            return
        raise NotImplementedError(f"Unsupported platform: {self._platform}")

    def load_case(self, case_spec: CaseSpec) -> dict[str, Any]:
        case = case_spec.normalized()
        case.validate()

        launch_mode: str
        note: str
        project_image_name: str | None = None

        if case.project_path:
            project_target = self._resolve_project_launch_target(case.project_path)
            if case.primary_image:
                project_image_name = self._resolve_project_image_name(project_target, case.primary_image)
            if project_image_name:
                self._launch_case_cli(project_target=project_target, image_name=project_image_name)
                launch_mode = "project_image"
                note = "Loaded the project and auto-selected the matching project image entry."
            else:
                self._launch_case_cli(project_target=project_target)
                launch_mode = "project"
                if case.primary_image:
                    note = (
                        "Loaded the project. The provided primary image was not auto-selected because no matching"
                        " project image entry could be resolved from the .qpproj file."
                    )
                else:
                    note = "Loaded the project."
            launch_target = project_target
        elif case.primary_image:
            self._launch_case_cli(image_path=case.primary_image)
            launch_mode = "single_image"
            note = "Loaded the primary image."
            launch_target = case.primary_image
        else:
            raise ValueError("No launch target could be determined from CaseSpec.")

        result = {
            "ok": True,
            "launch_mode": launch_mode,
            "launch_target": str(launch_target),
            "case": case.summary_dict(),
            "note": note,
        }
        if project_image_name:
            result["project_image_name"] = project_image_name
        return result

    def close_without_save(self) -> None:
        if self._platform == "darwin":
            macos.app_quit_without_save(self.process_name)
            stopped = self._wait_until_process_stops(timeout_seconds=max(5.0, self.config.load_wait_seconds))
            if not stopped:
                raise RuntimeError(
                    "QuPath did not exit after requesting close without save. "
                    "A save dialog may still be open."
                )
            self._target_pid = None
            return
        # Windows path is intentionally conservative in v1.

    @staticmethod
    def _resolve_project_launch_target(project_path: Path) -> Path:
        if project_path.is_file():
            return project_path
        candidates = sorted(project_path.glob("*.qpproj"))
        if candidates:
            return candidates[0]
        return project_path

    def _launch_case_cli(
        self,
        *,
        project_target: Path | None = None,
        image_path: Path | None = None,
        image_name: str | None = None,
    ) -> None:
        existing_pids = self._running_process_ids() if self._platform == "darwin" else []
        if self._platform == "darwin" and self._is_process_running():
            if self._running_process_requires_agent_restart():
                self.close_without_save()
                stopped = self._wait_until_process_stops(timeout_seconds=max(5.0, self.config.load_wait_seconds))
                if not stopped:
                    raise RuntimeError(
                        "Failed to restart the running QuPath process after rebuilding the attach helper. "
                        "Close QuPath manually and retry."
                    )
            if self._is_process_running():
                self._load_case_into_running_app(
                    project_target=project_target,
                    image_path=image_path,
                    image_name=image_name,
                )
                sleep_seconds(self.config.load_wait_seconds)
                try:
                    macos.activate_process(self.process_name)
                except Exception:
                    pass
                return

        cli_args = ["-q"]
        if project_target is not None:
            cli_args.extend(["-p", str(project_target)])
        if image_name:
            cli_args.extend(["-i", image_name])
        elif image_path is not None:
            cli_args.extend(["-i", str(image_path)])

        if self._platform == "darwin":
            app_bundle = macos.derive_app_bundle(self.config.executable_path)
            macos.open_with_app_args(app_bundle, cli_args)
        else:
            cmd = [str(self.config.executable_path), *cli_args]
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        sleep_seconds(self.config.load_wait_seconds)
        if self._platform == "darwin":
            self._select_target_pid(previous_pids=existing_pids)
            try:
                macos.activate_process(self.process_name)
            except Exception:
                pass
        elif self._platform == "windows":
            windows.activate_process(self.process_name)
        else:
            raise NotImplementedError(f"Unsupported platform: {self._platform}")

    def _load_case_into_running_app(
        self,
        *,
        project_target: Path | None = None,
        image_path: Path | None = None,
        image_name: str | None = None,
    ) -> None:
        pid = self._running_process_id()
        if pid is None:
            raise RuntimeError("QuPath is not running.")
        self._target_pid = pid

        helper_jar = self._ensure_running_qupath_agent_jar()
        helper_java = self._require_java_command("java")
        error_file = Path(tempfile.gettempdir()) / f"qupath_attach_error_{pid}.log"
        payload = urlencode(
            {
                key: value
                for key, value in {
                    "project": str(project_target) if project_target is not None else None,
                    "imageName": image_name,
                    "imagePath": str(image_path) if image_path is not None else None,
                    "errorPath": str(error_file),
                }.items()
                if value
            }
        )
        completed = subprocess.run(
            [
                helper_java,
                "--add-modules",
                "jdk.attach",
                "-jar",
                str(helper_jar),
                str(pid),
                payload,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            target_error = ""
            if error_file.exists():
                target_error = error_file.read_text(encoding="utf-8", errors="replace").strip()
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
            if target_error:
                detail = f"{detail}\n{target_error}"
            raise RuntimeError(f"Failed to load the case into the running QuPath instance.\n{detail}")

    def _is_process_running(self) -> bool:
        if self._platform != "darwin":
            return False
        return bool(self._running_process_ids())

    def _running_process_ids(self) -> list[int]:
        if self._platform != "darwin":
            return []
        completed = subprocess.run(
            ["pgrep", "-x", self.process_name],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return []
        pids: list[int] = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        return pids

    def _running_process_id(self) -> int | None:
        pids = self._running_process_ids()
        if not pids:
            self._target_pid = None
            return None
        if self._target_pid in pids:
            return self._target_pid
        self._target_pid = max(pids)
        return self._target_pid

    def _select_target_pid(self, *, previous_pids: list[int] | None = None) -> int | None:
        pids = self._running_process_ids()
        if not pids:
            self._target_pid = None
            return None
        previous = set(previous_pids or [])
        new_pids = [pid for pid in pids if pid not in previous]
        if new_pids:
            self._target_pid = max(new_pids)
            return self._target_pid
        if self._target_pid in pids:
            return self._target_pid
        self._target_pid = max(pids)
        return self._target_pid

    def running_process_start_time(self) -> datetime | None:
        pid = self._running_process_id()
        if pid is None:
            return None
        if self._platform == "darwin":
            completed = subprocess.run(
                ["ps", "-p", str(pid), "-o", "lstart="],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                return None
            raw = completed.stdout.strip()
            if not raw:
                return None
            try:
                return datetime.strptime(raw, "%a %b %d %H:%M:%S %Y")
            except ValueError:
                return None
        return None

    def _running_process_requires_agent_restart(self) -> bool:
        process_started_at = self.running_process_start_time()
        if process_started_at is None:
            return False
        try:
            helper_jar = self._ensure_running_qupath_agent_jar()
        except Exception:
            return False
        helper_built_at = datetime.fromtimestamp(helper_jar.stat().st_mtime)
        return helper_built_at > process_started_at

    def _wait_until_process_stops(self, *, timeout_seconds: float) -> bool:
        if self._platform != "darwin":
            return True
        remaining = max(0.0, float(timeout_seconds))
        interval = 0.25
        while remaining > 0:
            if not self._is_process_running():
                return True
            sleep_seconds(interval)
            remaining -= interval
        return not self._is_process_running()

    @staticmethod
    def _resolve_project_image_name(project_target: Path, image_path: Path) -> str | None:
        if not project_target.is_file() or project_target.suffix.lower() != ".qpproj":
            return None
        try:
            project = json.loads(project_target.read_text(encoding="utf-8"))
        except Exception:
            return None
        normalized_image = image_path.expanduser().resolve()
        for image in project.get("images", []):
            server = image.get("serverBuilder") or {}
            uri = str(server.get("uri") or "")
            resolved_path = QuPathLauncher._path_from_file_uri(uri)
            if resolved_path is not None and resolved_path == normalized_image:
                image_name = image.get("imageName")
                if isinstance(image_name, str) and image_name.strip():
                    return image_name
        return None

    @staticmethod
    def _path_from_file_uri(uri: str) -> Path | None:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or not parsed.path:
            return None
        return Path(unquote(parsed.path)).expanduser().resolve()

    def _ensure_running_qupath_agent_jar(self) -> Path:
        source_dir = Path(__file__).with_name("java")
        java_source_paths = [
            source_dir / "RunningQuPathAgent.java",
            source_dir / "RunningQuPathAttacher.java",
        ]
        build_dependency_paths = [*java_source_paths, Path(__file__)]
        missing = [path for path in java_source_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Running QuPath agent source not found: {missing[0]}")

        build_root = Path.home() / ".cache" / "medflow-bench" / "qupath_attach"
        classes_dir = build_root / "classes"
        jar_path = build_root / "running-qupath-agent.jar"
        manifest_path = build_root / "manifest.mf"
        expected_agent_class = "qupath_agent_engine.software.qupath.java.RunningQuPathAgent"
        expected_attacher_class = "qupath_agent_engine.software.qupath.java.RunningQuPathAttacher"

        newest_source_mtime = max(path.stat().st_mtime for path in build_dependency_paths)
        if (
            jar_path.exists()
            and jar_path.stat().st_mtime >= newest_source_mtime
            and self._running_qupath_agent_jar_matches_expected(
                jar_path,
                expected_agent_class=expected_agent_class,
                expected_attacher_class=expected_attacher_class,
            )
        ):
            return jar_path

        build_root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(classes_dir, ignore_errors=True)
        classes_dir.mkdir(parents=True, exist_ok=True)

        javac = self._require_java_command("javac")
        jar = self._require_java_command("jar")
        subprocess.run(
            [
                javac,
                "--add-modules",
                "jdk.attach",
                "-d",
                str(classes_dir),
                *(str(path) for path in java_source_paths),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        manifest_path.write_text(
            "\n".join(
                [
                    "Manifest-Version: 1.0",
                    f"Agent-Class: {expected_agent_class}",
                    f"Main-Class: {expected_attacher_class}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        subprocess.run(
            [
                jar,
                "--create",
                "--file",
                str(jar_path),
                "--manifest",
                str(manifest_path),
                "-C",
                str(classes_dir),
                ".",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return jar_path

    @staticmethod
    def _running_qupath_agent_jar_matches_expected(
        jar_path: Path,
        *,
        expected_agent_class: str,
        expected_attacher_class: str,
    ) -> bool:
        expected_entries = {
            f"{expected_agent_class.replace('.', '/')}.class",
            f"{expected_attacher_class.replace('.', '/')}.class",
        }
        try:
            with zipfile.ZipFile(jar_path) as jar_file:
                manifest = jar_file.read("META-INF/MANIFEST.MF").decode("utf-8", errors="replace")
                names = set(jar_file.namelist())
        except Exception:
            return False

        return (
            f"Agent-Class: {expected_agent_class}" in manifest
            and f"Main-Class: {expected_attacher_class}" in manifest
            and expected_entries.issubset(names)
        )

    @staticmethod
    def _require_java_command(name: str) -> str:
        command = shutil.which(name)
        if command:
            return command
        raise RuntimeError(f"Required Java tool not found on PATH: {name}")
