from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from qupath_agent_engine.core.utils import compact

from .config import QuPathConfig
from .launcher import QuPathLauncher


QUPATH_IMAGE_TYPE_OPTIONS: dict[str, str] = {
    "BRIGHTFIELD_H_DAB": "Brightfield (H-DAB)",
    "BRIGHTFIELD_H_E": "Brightfield (H&E)",
    "BRIGHTFIELD_OTHER": "Brightfield (other)",
    "FLUORESCENCE": "Fluorescence",
    "OTHER": "Other",
    "UNSET": "Not set",
}

VIEWER_COORD_MIN = 0.0
VIEWER_COORD_MAX = 999.0
VIEWER_COORD_RANGE_TEXT = "0..999"

ZOOM_DIRECTION_OPTIONS = ("in", "out")
ZOOM_UNIT_OPTIONS = ("magnification", "relative_factor", "steps")
ZOOM_UNIT_ALIASES = {
    "magnification": "magnification",
    "target_magnification": "magnification",
    "absolute": "magnification",
    "relative_factor": "relative_factor",
    "factor": "relative_factor",
    "relative": "relative_factor",
    "multiplier": "relative_factor",
    "steps": "steps",
}
PAN_DIRECTION_OPTIONS = (
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)
PAN_MODE_OPTIONS = ("directional", "set_center")
PAN_MODE_ALIASES = {
    "direction": "directional",
    "directional": "directional",
    "relative": "directional",
    "move": "directional",
    "set_center": "set_center",
    "center": "set_center",
    "recenter": "set_center",
    "target_center": "set_center",
    "viewer_point": "set_center",
}
PAN_UNIT_OPTIONS = ("viewport_fraction", "image_pixels")


class QuPathDirectController:
    """Direct, non-GUI control for a running QuPath instance.

    This reuses the existing Java agent attach path that the launcher already uses
    for loading a case into a running QuPath instance. The goal is to keep these
    tool actions deterministic and separate from Midscene.
    """

    def __init__(self, *, config: QuPathConfig, launcher: QuPathLauncher) -> None:
        self.config = config.normalized()
        self.launcher = launcher
        self._platform = platform.system().lower()

    def query_state(self) -> dict[str, Any]:
        return self._invoke_action("query_state", {})

    def discard_unsaved_changes(self) -> dict[str, Any]:
        if not self._can_attach_to_running_qupath():
            return {
                "ok": False,
                "error": (
                    "Discarding unsaved QuPath changes requires a running QuPath instance that can be reached "
                    "through the Java attach bridge."
                ),
            }
        return self._invoke_action("discard_unsaved_changes", {})

    def score_point_against_qpdata(self, *, qpdata_path: str | Path, x: float, y: float) -> dict[str, Any]:
        qpdata = Path(qpdata_path).expanduser().resolve()
        if not qpdata.exists():
            return {
                "ok": False,
                "error": f"Qpdata path does not exist: {qpdata}",
            }
        return self._invoke_action(
            "score_point_against_qpdata",
            {
                "qpdataPath": str(qpdata),
                "x": float(x),
                "y": float(y),
            },
        )

    def open_project(self, *, project_path: str | Path, image_name: str | None = None) -> dict[str, Any]:
        project = Path(project_path).expanduser().resolve()
        if not project.exists():
            return {
                "ok": False,
                "error": f"Project path does not exist: {project}",
            }
        resolved_project = self.launcher._resolve_project_launch_target(project)
        if self._can_attach_to_running_qupath():
            return self._invoke_action(
                "open_project",
                {
                    "project": str(resolved_project),
                    "imageName": image_name,
                },
            )

        if image_name:
            return {
                "ok": False,
                "error": (
                    "Direct attach is unavailable, and opening a specific project image entry by exact image name "
                    "requires an attached running QuPath instance."
                ),
                "project_path": str(resolved_project),
                "image_name": image_name,
            }

        # Conservative fallback: reuse the existing, known-good launch/load path.
        from qupath_agent_engine.core.types import CaseSpec

        result = self.launcher.load_case(CaseSpec(project_path=resolved_project).normalized())
        return {
            "ok": bool(result.get("ok", True)),
            "action": "open_project",
            "mode": "launcher_fallback",
            "project_path": str(resolved_project),
            "result": result,
        }

    def load_image(self, *, image_path: str | Path) -> dict[str, Any]:
        image = Path(image_path).expanduser().resolve()
        if not image.exists():
            return {
                "ok": False,
                "error": f"Image path does not exist: {image}",
            }
        if self._can_attach_to_running_qupath():
            return self._invoke_action(
                "load_image",
                {
                    "imagePath": str(image),
                },
            )

        from qupath_agent_engine.core.types import CaseSpec

        result = self.launcher.load_case(CaseSpec(image_paths=[image]).normalized())
        return {
            "ok": bool(result.get("ok", True)),
            "action": "load_image",
            "mode": "launcher_fallback",
            "image_path": str(image),
            "result": result,
        }

    def set_image_type(self, *, image_type: str) -> dict[str, Any]:
        normalized = str(image_type or "").strip().upper()
        if normalized not in QUPATH_IMAGE_TYPE_OPTIONS:
            return {
                "ok": False,
                "error": (
                    f"Unsupported image type: {image_type}. Supported values: "
                    f"{', '.join(QUPATH_IMAGE_TYPE_OPTIONS.keys())}"
                ),
            }
        if not self._can_attach_to_running_qupath():
            return {
                "ok": False,
                "error": (
                    "Direct image-type control requires a running QuPath instance that can be reached through the Java attach bridge."
                ),
                "requested_image_type": normalized,
            }
        return self._invoke_action(
            "set_image_type",
            {
                "imageType": normalized,
            },
        )

    def zoom(
        self,
        *,
        direction: str,
        amount: float,
        unit: str,
        x: float | None = None,
        y: float | None = None,
    ) -> dict[str, Any]:
        normalized_direction = str(direction or "").strip().lower()
        normalized_unit = normalize_zoom_unit(unit)
        try:
            numeric_amount = float(amount)
        except Exception:
            return {
                "ok": False,
                "error": "Zoom amount must be numeric.",
                "amount": amount,
                "unit": unit,
            }

        viewer_point, viewer_point_error = _parse_optional_viewer_point(x=x, y=y)
        if viewer_point_error is not None:
            return viewer_point_error

        if normalized_unit is None:
            return {
                "ok": False,
                "error": (
                    f"Unsupported zoom unit: {unit}. Supported values: {', '.join(ZOOM_UNIT_OPTIONS)}"
                ),
            }

        if normalized_unit == "magnification":
            if numeric_amount <= 0.0:
                return {
                    "ok": False,
                    "error": "Zoom amount must be > 0 when unit='magnification'.",
                }
        else:
            if normalized_direction not in ZOOM_DIRECTION_OPTIONS:
                return {
                    "ok": False,
                    "error": (
                        f"Unsupported zoom direction: {direction}. Supported values: {', '.join(ZOOM_DIRECTION_OPTIONS)}"
                    ),
                }
            if normalized_unit == "relative_factor" and numeric_amount <= 1.0:
                return {
                    "ok": False,
                    "error": "Zoom amount must be > 1.0 when unit='relative_factor'.",
                }
            if normalized_unit == "steps" and numeric_amount < 1:
                return {
                    "ok": False,
                    "error": "Zoom amount must be >= 1 when unit='steps'.",
                }

        if not self._can_attach_to_running_qupath():
            return {
                "ok": False,
                "error": "Direct zoom requires a running QuPath instance that can be reached through the Java attach bridge.",
                "direction": normalized_direction,
                "amount": numeric_amount,
                "unit": normalized_unit,
                **(viewer_point or {}),
            }

        return self._invoke_action(
            "zoom",
            {
                "direction": normalized_direction,
                "amount": numeric_amount,
                "unit": normalized_unit,
                **(viewer_point or {}),
            },
        )

    def pan(
        self,
        *,
        direction: str | None = None,
        amount: float | None = None,
        unit: str | None = None,
        mode: str | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> dict[str, Any]:
        viewer_point, viewer_point_error = _parse_optional_viewer_point(x=x, y=y)
        if viewer_point_error is not None:
            return viewer_point_error

        normalized_mode = normalize_pan_mode(mode)
        if str(mode or "").strip() and normalized_mode is None:
            return {
                "ok": False,
                "error": f"Unsupported pan mode: {mode}. Supported values: {', '.join(PAN_MODE_OPTIONS)}",
            }
        if normalized_mode is None:
            normalized_mode = "set_center" if viewer_point is not None else "directional"

        numeric_amount: float | None = None
        normalized_direction = str(direction or "").strip().lower()
        normalized_unit = str(unit or "").strip().lower()
        if normalized_mode == "set_center":
            if viewer_point is None:
                return {
                    "ok": False,
                    "error": (
                        "Pan mode 'set_center' requires x and y on the current viewer screenshot "
                        f"in the range {VIEWER_COORD_RANGE_TEXT}."
                    ),
                    "mode": normalized_mode,
                }
        else:
            if normalized_direction not in PAN_DIRECTION_OPTIONS:
                return {
                    "ok": False,
                    "error": f"Unsupported pan direction: {direction}. Supported values: {', '.join(PAN_DIRECTION_OPTIONS)}",
                }
            if normalized_unit not in PAN_UNIT_OPTIONS:
                return {
                    "ok": False,
                    "error": f"Unsupported pan unit: {unit}. Supported values: {', '.join(PAN_UNIT_OPTIONS)}",
                }
            try:
                numeric_amount = float(amount)
            except Exception:
                return {
                    "ok": False,
                    "error": "Pan amount must be numeric when using directional pan.",
                    "amount": amount,
                }
            if numeric_amount <= 0:
                return {
                    "ok": False,
                    "error": "Pan amount must be > 0.",
                }
        if not self._can_attach_to_running_qupath():
            return {
                "ok": False,
                "error": "Direct pan requires a running QuPath instance that can be reached through the Java attach bridge.",
                "mode": normalized_mode,
                "direction": normalized_direction,
                "amount": numeric_amount,
                "unit": normalized_unit,
                **(viewer_point or {}),
            }
        return self._invoke_action(
            "pan",
            {
                "mode": normalized_mode,
                "direction": normalized_direction,
                "amount": numeric_amount,
                "unit": normalized_unit,
                **(viewer_point or {}),
            },
        )

    def viewer_point_to_wsi_coordinate(self, *, x: float, y: float) -> dict[str, Any]:
        viewer_point, viewer_point_error = _parse_required_viewer_point(x=x, y=y)
        if viewer_point_error is not None:
            return viewer_point_error
        assert viewer_point is not None
        if not self._can_attach_to_running_qupath():
            return {
                "ok": False,
                "error": (
                    "Direct viewer-point conversion requires a running QuPath instance that can be reached through the Java attach bridge."
                ),
                **viewer_point,
            }
        return self._invoke_action(
            "viewer_point_to_wsi_coordinate",
            viewer_point,
        )

    def _can_attach_to_running_qupath(self) -> bool:
        if self._platform not in {"darwin", "windows"}:
            return False
        try:
            return self._running_process_id() is not None
        except Exception:
            return False

    def _invoke_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        pid = self._running_process_id()
        if pid is None:
            return {
                "ok": False,
                "action": action,
                "error": "QuPath is not running. Start or load QuPath first.",
            }

        if self.launcher._running_process_requires_agent_restart():
            return {
                "ok": False,
                "action": action,
                "error": (
                    "The running QuPath process was started before the current attach helper was built. "
                    "Restart QuPath or reload the case so the direct-control agent uses the current code."
                ),
            }

        helper_jar = self.launcher._ensure_running_qupath_agent_jar()
        helper_java = self.launcher._require_java_command("java")

        output_fd, output_name = tempfile.mkstemp(suffix=f"_{action}_output.json")
        os.close(output_fd)
        output_path = Path(output_name)
        error_fd, error_name = tempfile.mkstemp(suffix=f"_{action}_error.log")
        os.close(error_fd)
        error_path = Path(error_name)

        payload = urlencode(
            {
                key: value
                for key, value in {
                    "action": action,
                    **params,
                    "outputPath": str(output_path),
                    "errorPath": str(error_path),
                }.items()
                if value is not None and value != ""
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

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        error_text = ""
        if error_path.exists():
            try:
                error_text = error_path.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                error_text = ""
        output_data: dict[str, Any] | None = None
        if output_path.exists():
            try:
                parsed = json.loads(output_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    output_data = parsed
            except Exception:
                output_data = None

        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            error_path.unlink(missing_ok=True)
        except Exception:
            pass

        if completed.returncode != 0 or error_text:
            return {
                **(output_data or {}),
                "ok": False,
                "action": action,
                "error": error_text or stderr.strip() or stdout.strip() or f"attach bridge failed with exit code {completed.returncode}",
                "stdout": compact(stdout, max_chars=4000),
                "stderr": compact(stderr, max_chars=4000),
            }

        if output_data is None:
            return {
                "ok": False,
                "action": action,
                "error": "The attach bridge returned no structured output.",
                "stdout": compact(stdout, max_chars=4000),
                "stderr": compact(stderr, max_chars=4000),
            }

        output_data.setdefault("ok", True)
        output_data.setdefault("action", action)
        output_data["stdout"] = compact(stdout, max_chars=4000)
        output_data["stderr"] = compact(stderr, max_chars=4000)
        return output_data

    def _running_process_id(self) -> int | None:
        if self._platform == "darwin":
            return self.launcher._running_process_id()

        if self._platform == "windows":
            executable_path = str(self.config.executable_path)
            process_name_candidates = {
                self.launcher.process_name,
                Path(self.launcher.process_name).stem,
                self.config.executable_path.name,
                self.config.executable_path.stem,
            }
            executable_path_escaped = executable_path.replace("'", "''")
            name_checks = " or ".join(
                "$_.Name -eq '{}'".format(str(name).replace("'", "''"))
                for name in sorted(candidate for candidate in process_name_candidates if candidate)
            ) or "$false"
            ps_script = (
                "$p = Get-CimInstance Win32_Process | Where-Object { "
                f"$_.ExecutablePath -eq '{executable_path_escaped}'"
                f" or {name_checks} "
                "} | Select-Object -First 1 -ExpandProperty ProcessId;"
                "if ($p) { Write-Output $p }"
            )
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_script],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                return None
            for line in completed.stdout.splitlines():
                line = line.strip()
                if line.isdigit():
                    return int(line)
            return None

        return None


def normalize_zoom_unit(unit: str | None) -> str | None:
    raw = str(unit or "").strip().lower()
    return ZOOM_UNIT_ALIASES.get(raw)


def normalize_pan_mode(mode: str | None) -> str | None:
    raw = str(mode or "").strip().lower()
    if not raw:
        return None
    return PAN_MODE_ALIASES.get(raw)


def _parse_required_viewer_point(*, x: float, y: float) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    return _parse_viewer_point_pair(x=x, y=y, allow_missing_pair=False)


def _parse_optional_viewer_point(*, x: float | None, y: float | None) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    return _parse_viewer_point_pair(x=x, y=y, allow_missing_pair=True)


def _parse_viewer_point_pair(
    *,
    x: float | None,
    y: float | None,
    allow_missing_pair: bool,
) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    if x is None and y is None:
        if allow_missing_pair:
            return None, None
        return None, {
            "ok": False,
            "error": f"x and y are required and must each be between {int(VIEWER_COORD_MIN)} and {int(VIEWER_COORD_MAX)}, inclusive.",
        }
    if x is None or y is None:
        return None, {
            "ok": False,
            "error": "x and y must either both be provided or both be omitted.",
            "x": x,
            "y": y,
        }
    try:
        numeric_x = float(x)
        numeric_y = float(y)
    except Exception:
        return None, {
            "ok": False,
            "error": "x and y must be numeric.",
            "x": x,
            "y": y,
        }
    if not (VIEWER_COORD_MIN <= numeric_x <= VIEWER_COORD_MAX) or not (VIEWER_COORD_MIN <= numeric_y <= VIEWER_COORD_MAX):
        return None, {
            "ok": False,
            "error": f"x and y must each be between {int(VIEWER_COORD_MIN)} and {int(VIEWER_COORD_MAX)}, inclusive.",
            "x": numeric_x,
            "y": numeric_y,
        }
    return {"x": numeric_x, "y": numeric_y}, None


def extract_rule_based_gui_state(state_payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(state_payload, dict):
        return {"ok": False, "error": "Rule-based QuPath GUI state unavailable."}
    if not state_payload.get("ok"):
        return {
            "ok": False,
            "error": _clean_text(state_payload.get("error")) or "Rule-based QuPath GUI state unavailable.",
        }
    state = state_payload.get("state") or {}
    if not isinstance(state, dict):
        return {"ok": False, "error": "Rule-based QuPath GUI state unavailable."}
    gui = state.get("gui") or {}
    if not isinstance(gui, dict):
        gui = {}
    return {
        "ok": True,
        "result": gui,
    }


def render_qupath_state_text(state_payload: dict[str, Any], *, redact_sensitive: bool = False) -> str:
    if not isinstance(state_payload, dict):
        return "QuPath direct state unavailable."
    if not state_payload.get("ok"):
        error = state_payload.get("error")
        if isinstance(error, str) and error.strip():
            return f"QuPath direct state unavailable: {error.strip()}"
        return "QuPath direct state unavailable."

    state = state_payload.get("state") or {}
    if not isinstance(state, dict):
        return "QuPath direct state unavailable."

    lines: list[str] = []
    project_name = _clean_text(state.get("project_name"))
    project_path = _clean_text(state.get("project_path"))
    displayed_image_name = _clean_text(state.get("displayed_image_name"))
    server_path = _clean_text(state.get("server_path"))
    image_type_value = _clean_text(state.get("image_type_value"))
    image_type_label = _clean_text(state.get("image_type_label"))
    if redact_sensitive:
        lines.append(f"Project loaded: {_yes_no(bool(project_name or project_path))}")
        lines.append(f"Displayed image loaded: {_yes_no(bool(displayed_image_name or server_path))}")
    else:
        lines.append(f"Project: {project_name or 'none'}")
        if project_path:
            lines.append(f"Project path: {project_path}")
        lines.append(f"Displayed image: {displayed_image_name or 'none'}")
        if server_path:
            lines.append(f"Server path: {server_path}")
    if image_type_value or image_type_label:
        if image_type_value and image_type_label:
            lines.append(f"Image type: {image_type_value} ({image_type_label})")
        else:
            lines.append(f"Image type: {image_type_value or image_type_label}")

    viewer = state.get("viewer") or {}
    if isinstance(viewer, dict):
        cx = viewer.get("center_pixel_x")
        cy = viewer.get("center_pixel_y")
        downsample = viewer.get("downsample")
        magnification = viewer.get("magnification")
        full_magnification = viewer.get("full_magnification")
        if cx is not None and cy is not None:
            lines.append(f"Viewer center: ({_fmt(cx)}, {_fmt(cy)})")
        if downsample is not None:
            lines.append(f"Viewer downsample: {_fmt(downsample)}")
        if magnification is not None:
            if isinstance(full_magnification, (int, float)):
                lines.append(
                    f"Viewer magnification: {_fmt(magnification)}x (full-resolution metadata: {_fmt(full_magnification)}x)"
                )
            else:
                lines.append(f"Viewer magnification: {_fmt(magnification)}x")
        region = viewer.get("displayed_region") or {}
        if isinstance(region, dict) and region:
            lines.append(
                "Visible region: "
                f"x={_fmt(region.get('x'))}, y={_fmt(region.get('y'))}, "
                f"w={_fmt(region.get('width'))}, h={_fmt(region.get('height'))}"
            )
        image_width = viewer.get("image_width")
        image_height = viewer.get("image_height")
        if image_width is not None and image_height is not None:
            lines.append(f"Image size: {_fmt(image_width)} x {_fmt(image_height)}")

    return "\n".join(lines).strip() or "QuPath direct state unavailable."


def render_qupath_gui_state_text(gui_payload: dict[str, Any], *, redact_sensitive: bool = False) -> str:
    if not isinstance(gui_payload, dict):
        return "Rule-based QuPath GUI state unavailable."
    if not gui_payload.get("ok"):
        error = _clean_text(gui_payload.get("error"))
        return f"Rule-based QuPath GUI state unavailable: {error}" if error else "Rule-based QuPath GUI state unavailable."

    gui = gui_payload.get("result") or {}
    if not isinstance(gui, dict):
        return "Rule-based QuPath GUI state unavailable."

    lines: list[str] = []
    app_version = _clean_text(gui.get("app_version"))
    window_title = _clean_text(gui.get("window_title"))
    if app_version:
        lines.append(f"App version: {app_version}")
    if window_title and not redact_sensitive:
        lines.append(f"Window title: {window_title}")

    read_only = gui.get("read_only")
    if read_only is not None:
        lines.append(f"Read-only mode: {_yes_no(read_only)}")
    ui_blocked = gui.get("ui_blocked")
    if ui_blocked is not None:
        lines.append(f"UI blocked: {_yes_no(ui_blocked)}")
    analysis_pane_visible = gui.get("analysis_pane_visible")
    if analysis_pane_visible is not None:
        lines.append(f"Analysis pane visible: {_yes_no(analysis_pane_visible)}")
    input_display_visible = gui.get("input_display_visible")
    if input_display_visible is not None:
        lines.append(f"Input display visible: {_yes_no(input_display_visible)}")

    selected_tool = _clean_text(gui.get("selected_tool_name"))
    previous_tool = _clean_text(gui.get("previous_tool_name"))
    if selected_tool:
        lines.append(f"Selected tool: {selected_tool}")
    if previous_tool:
        lines.append(f"Previous tool: {previous_tool}")
    tool_switching_enabled = gui.get("tool_switching_enabled")
    if tool_switching_enabled is not None:
        lines.append(f"Tool switching enabled: {_yes_no(tool_switching_enabled)}")
    selected_tool_locked = gui.get("selected_tool_locked")
    if selected_tool_locked is not None:
        lines.append(f"Selected tool locked: {_yes_no(selected_tool_locked)}")

    analysis_tabs = gui.get("analysis_tabs") or []
    if isinstance(analysis_tabs, list) and analysis_tabs:
        tab_names = [_clean_text(tab.get("title") if isinstance(tab, dict) else tab) for tab in analysis_tabs]
        tab_names = [name for name in tab_names if name]
        selected_tab = _clean_text(gui.get("selected_analysis_tab"))
        if tab_names:
            if selected_tab:
                lines.append(f"Analysis tabs: {', '.join(tab_names)} (selected: {selected_tab})")
            else:
                lines.append(f"Analysis tabs: {', '.join(tab_names)}")

    toolbar_items = gui.get("toolbar_items") or []
    toolbar_labels = _toolbar_labels(toolbar_items)
    if toolbar_labels:
        lines.append(f"Toolbar items: {', '.join(toolbar_labels[:14])}")
        if len(toolbar_labels) > 14:
            lines.append(f"Toolbar items omitted: {len(toolbar_labels) - 14} more")

    available_tools = gui.get("available_tools") or []
    available_tool_names = _available_tool_names(available_tools)
    if available_tool_names:
        lines.append(f"Available drawing/viewer tools: {', '.join(available_tool_names[:12])}")
        if len(available_tool_names) > 12:
            lines.append(f"Available tools omitted: {len(available_tool_names) - 12} more")

    return "\n".join(lines).strip() or "Rule-based QuPath GUI state unavailable."


def _toolbar_labels(toolbar_items: list[Any]) -> list[str]:
    labels: list[str] = []
    for item in toolbar_items:
        if isinstance(item, dict):
            label = _clean_text(item.get("label"))
        else:
            label = _clean_text(item)
        if label:
            labels.append(label)
    return labels


def _available_tool_names(available_tools: list[Any]) -> list[str]:
    names: list[str] = []
    for item in available_tools:
        if isinstance(item, dict):
            name = _clean_text(item.get("name"))
        else:
            name = _clean_text(item)
        if name:
            names.append(name)
    return names


def _yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return _clean_text(value) or "unknown"


def _fmt(value: Any) -> str:
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.3f}"
    if value is None:
        return "?"
    return str(value)


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()
