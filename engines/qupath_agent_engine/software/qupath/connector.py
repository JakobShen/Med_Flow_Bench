from __future__ import annotations

from pathlib import Path
from typing import Any

from qupath_agent_engine.core.run_context import RunContext
from qupath_agent_engine.core.types import CaseSpec, StepObservation
from qupath_agent_engine.core.utils import sanitize_filename, sleep_seconds
from qupath_agent_engine.software.base import SoftwareConnector, ToolHandler

from .capture import ViewerCapture
from .config import QuPathConfig
from .direct_control import (
    QuPathDirectController,
    extract_rule_based_gui_state,
    render_qupath_gui_state_text,
    render_qupath_state_text,
)
from .launcher import QuPathLauncher
from .midscene_client import MidsceneBridge


class QuPathConnector(SoftwareConnector):
    def __init__(
        self,
        *,
        config: QuPathConfig,
        run_context: RunContext,
        case_spec: CaseSpec,
        redact_model_state: bool = False,
    ) -> None:
        self.config = config.normalized()
        self.run_context = run_context
        self.case_spec = case_spec.normalized()
        self.redact_model_state = bool(redact_model_state)
        self.launcher = QuPathLauncher(self.config)
        self.capture = ViewerCapture(config=self.config, launcher=self.launcher)
        self.direct = QuPathDirectController(config=self.config, launcher=self.launcher)
        self.midscene = MidsceneBridge(self.config)
        self.recorded_evidence: list[dict[str, Any]] = []
        self.localization_queries: list[dict[str, Any]] = []

    def bootstrap(self) -> dict[str, Any]:
        result = self.launcher.launch_app_only()
        self.run_context.logger.event("qupath_bootstrap", result=result)
        return result

    def load_case(self) -> dict[str, Any]:
        result = self.launcher.load_case(self.case_spec)
        self.run_context.logger.event("qupath_load_case", result=result)
        return result

    def open_project(self, *, project_path: str, image_name: str | None = None) -> dict[str, Any]:
        before_state = self.query_qupath_state()
        raw_result = self.direct.open_project(project_path=project_path, image_name=image_name)
        after_state = self.query_qupath_state()
        raw_payload = {
            **raw_result,
            "state_before": before_state.get("state") if isinstance(before_state, dict) else None,
            "state_after": after_state.get("state") if isinstance(after_state, dict) else None,
        }
        host_payload = self._host_visible_direct_result("open_project", raw_payload, redact_sensitive=self.redact_model_state)
        self.run_context.logger.event(
            "qupath_open_project",
            project_path=project_path,
            image_name=image_name,
            result_host=host_payload,
            result_raw=raw_payload,
        )
        return host_payload

    def load_image(self, *, image_path: str) -> dict[str, Any]:
        before_state = self.query_qupath_state()
        raw_result = self.direct.load_image(image_path=image_path)
        after_state = self.query_qupath_state()
        raw_payload = {
            **raw_result,
            "state_before": before_state.get("state") if isinstance(before_state, dict) else None,
            "state_after": after_state.get("state") if isinstance(after_state, dict) else None,
        }
        host_payload = self._host_visible_direct_result("load_image", raw_payload, redact_sensitive=self.redact_model_state)
        self.run_context.logger.event(
            "qupath_load_image",
            image_path=image_path,
            result_host=host_payload,
            result_raw=raw_payload,
        )
        return host_payload

    def set_image_type(self, *, image_type: str) -> dict[str, Any]:
        raw_result = self.direct.set_image_type(image_type=image_type)
        host_payload = self._host_visible_direct_result("set_image_type", raw_result, redact_sensitive=self.redact_model_state)
        self.run_context.logger.event(
            "qupath_set_image_type",
            image_type=image_type,
            result_host=host_payload,
            result_raw=raw_result,
        )
        return host_payload

    def zoom(
        self,
        *,
        direction: str,
        amount: float,
        unit: str,
        x: float | None = None,
        y: float | None = None,
    ) -> dict[str, Any]:
        raw_result = self.direct.zoom(direction=direction, amount=amount, unit=unit, x=x, y=y)
        host_payload = self._host_visible_direct_result("zoom", raw_result, redact_sensitive=self.redact_model_state)
        self.run_context.logger.event(
            "qupath_zoom",
            direction=direction,
            amount=amount,
            unit=unit,
            x=x,
            y=y,
            result_host=host_payload,
            result_raw=raw_result,
        )
        return host_payload

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
        raw_result = self.direct.pan(direction=direction, amount=amount, unit=unit, mode=mode, x=x, y=y)
        host_payload = self._host_visible_direct_result("pan", raw_result, redact_sensitive=self.redact_model_state)
        self.run_context.logger.event(
            "qupath_pan",
            mode=mode,
            direction=direction,
            amount=amount,
            unit=unit,
            x=x,
            y=y,
            result_host=host_payload,
            result_raw=raw_result,
        )
        return host_payload

    def query_qupath_state(self) -> dict[str, Any]:
        result = self.direct.query_state()
        self.run_context.logger.event("qupath_direct_state", result=result)
        return result

    def score_point_against_qpdata(self, *, qpdata_path: str | Path, x: float, y: float) -> dict[str, Any]:
        resolved_qpdata_path = str(Path(qpdata_path).expanduser().resolve())
        result = self.direct.score_point_against_qpdata(qpdata_path=qpdata_path, x=x, y=y)
        self.run_context.logger.event(
            "qpdata_point_score",
            qpdata_path=resolved_qpdata_path,
            x=float(x),
            y=float(y),
            result=result,
        )
        if self.run_context.benchmark_logger is not None:
            self.run_context.benchmark_logger.event(
                "qpdata_point_score",
                qpdata_path=resolved_qpdata_path,
                x=float(x),
                y=float(y),
                ok=bool(isinstance(result, dict) and result.get("ok")),
                hit_any=bool(isinstance(result, dict) and result.get("hit_any")),
                n_hits=result.get("n_hits") if isinstance(result, dict) else None,
                annotation_summary=(
                    result.get("annotation_summary")
                    if isinstance(result, dict)
                    else None
                ),
                error=result.get("error") if isinstance(result, dict) else None,
            )
        return result

    def focus_qupath(self) -> dict[str, Any]:
        self.launcher.focus()
        result = {"ok": True, "focused": True, "process_name": self.launcher.process_name}
        self.run_context.logger.event("qupath_focus", result=result)
        return result

    def close_qupath(self, *, save: bool = False) -> dict[str, Any]:
        if save:
            result = {
                "ok": False,
                "error": "v1 deliberately does not implement save-on-close. Close without save only.",
            }
            self.run_context.logger.event("qupath_close", result=result)
            return result
        discard_result = self.direct.discard_unsaved_changes()
        self.run_context.logger.event("qupath_discard_unsaved_changes_before_close", result=discard_result)
        self.launcher.close_without_save()
        result = {"ok": True, "closed": True, "saved": False, "discard_unsaved_changes": discard_result}
        self.run_context.logger.event("qupath_close", result=result)
        return result

    def sleep(self, *, seconds: float) -> dict[str, Any]:
        sleep_seconds(seconds)
        result = {"ok": True, "slept_seconds": float(seconds)}
        self.run_context.logger.event("sleep", result=result)
        return result

    def capture_viewer(self, *, label: str | None = None) -> dict[str, Any]:
        safe = sanitize_filename(label or "viewer")
        capture_id = self.run_context.next_capture_id()
        filename = f"step_{capture_id:04d}_{safe}.png"
        output_path = self.run_context.evidence_path(filename)
        result = self.capture.capture_current_viewer(output_path)
        self.run_context.logger.event("capture_viewer", label=safe, result=result)
        return result

    def viewer_point_to_wsi_coordinate(self, *, x: float, y: float) -> dict[str, Any]:
        raw_result = self.direct.viewer_point_to_wsi_coordinate(x=x, y=y)
        host_payload = self._host_visible_direct_result(
            "viewer_point_to_wsi_coordinate",
            raw_result,
            redact_sensitive=self.redact_model_state,
        )
        localization_query = {
            "viewer_x": float(x),
            "viewer_y": float(y),
            "result": host_payload,
        }
        self.localization_queries.append(localization_query)
        self.run_context.summary["localization_queries"] = list(self.localization_queries)
        self.run_context.logger.event(
            "qupath_viewer_point_to_wsi_coordinate",
            viewer_x=float(x),
            viewer_y=float(y),
            result_host=host_payload,
            result_raw=raw_result,
        )
        return host_payload

    def record_current_view_as_evidence(self, *, supported_label: str | None = None, description: str | None = None) -> dict[str, Any]:
        normalized_label = str(supported_label or "").strip().upper() or None
        description_text = str(description or "").strip() or None

        qupath_state = self.query_qupath_state()
        if not isinstance(qupath_state, dict) or not qupath_state.get("ok"):
            error = qupath_state.get("error") if isinstance(qupath_state, dict) else None
            return {
                "ok": False,
                "error": error or "Unable to query the current QuPath state.",
                "text_output": f"record_current_view_as_evidence failed: {error or 'state unavailable.'}",
            }

        state = qupath_state.get("state") or {}
        viewer = _viewer_state(state)
        center = _center_tuple(viewer)
        if center is None:
            return {
                "ok": False,
                "error": "Viewer center is unavailable.",
                "text_output": "record_current_view_as_evidence failed: viewer center is unavailable.",
            }

        evidence_index = len(self.recorded_evidence) + 1
        filename = f"evidence_{evidence_index:04d}_{sanitize_filename(normalized_label or 'evidence', fallback='evidence')}.png"
        output_path = self.run_context.evidence_path(filename)
        capture_result = self.capture.capture_current_viewer(output_path)
        if not capture_result.get("ok"):
            error = capture_result.get("error") or "Viewer capture failed."
            self.run_context.logger.event(
                "record_current_view_as_evidence_error",
                supported_label=normalized_label,
                description=description_text,
                qupath_state=qupath_state,
                capture_result=capture_result,
            )
            return {
                "ok": False,
                "error": error,
                "text_output": f"record_current_view_as_evidence failed: {error}",
            }

        displayed_region = viewer.get("displayed_region") if isinstance(viewer.get("displayed_region"), dict) else {}
        evidence_record = {
            "evidence_id": f"evidence_{evidence_index:04d}",
            "supported_label": normalized_label,
            "description": description_text,
            "center_pixel_x": center[0],
            "center_pixel_y": center[1],
            "magnification": viewer.get("magnification"),
            "downsample": viewer.get("downsample"),
            "displayed_region": dict(displayed_region),
            "image_width": viewer.get("image_width"),
            "image_height": viewer.get("image_height"),
            "capture": dict(capture_result),
            "capture_path": str(output_path),
            "qupath_state": state,
        }
        self.recorded_evidence.append(evidence_record)
        self.run_context.summary["recorded_evidence"] = list(self.recorded_evidence)
        self.run_context.logger.event("record_current_view_as_evidence", evidence=evidence_record)

        center_text = _center_text(viewer)
        mag = viewer.get("magnification")
        mag_text = f" Magnification: {_fmt_num(mag)}x." if mag is not None else ""
        label_text = f" Supported label: {normalized_label}." if normalized_label else ""
        description_part = f" Description: {description_text}." if description_text else ""
        text_output = (
            "Evidence recorded."
            f"{label_text}"
            f"{description_part} "
            f"Current viewer center: {center_text}."
            f"{mag_text} "
            "The benchmark will judge this evidence using the CURRENT viewer center point. "
            "Ensure that center is inside the ROI you intend to cite."
        ).strip()
        return {
            "ok": True,
            "action": "record_current_view_as_evidence",
            "evidence_id": evidence_record["evidence_id"],
            "supported_label": normalized_label,
            "description": description_text,
            "center": {
                "x": center[0],
                "y": center[1],
            },
            "magnification": viewer.get("magnification"),
            "text_output": text_output,
        }

    def qupath_gui_act(self, *, instruction: str) -> dict[str, Any]:
        step_id = self.run_context.next_step_id()
        report_path = self.run_context.midscene_path(f"{step_id}_act.html")
        self.run_context.logger.event("midscene_act_start", instruction=instruction, step_id=step_id)
        result = self.midscene.act(
            instruction=instruction,
            report_path=report_path,
            step_log_file=self._shared_step_log_path(),
        ).payload
        self.run_context.logger.event("midscene_act_done", step_id=step_id, result=result)
        return result

    def qupath_gui_query(self, *, question: str) -> dict[str, Any]:
        step_id = self.run_context.next_step_id()
        report_path = self.run_context.midscene_path(f"{step_id}_query.html")
        self.run_context.logger.event("midscene_query_start", question=question, step_id=step_id)
        result = self.midscene.query(
            question=question,
            report_path=report_path,
            step_log_file=self._shared_step_log_path(),
        ).payload
        self.run_context.logger.event("midscene_query_done", step_id=step_id, result=result)
        return result

    def qupath_gui_assert(self, *, assertion: str) -> dict[str, Any]:
        step_id = self.run_context.next_step_id()
        report_path = self.run_context.midscene_path(f"{step_id}_assert.html")
        self.run_context.logger.event("midscene_assert_start", assertion=assertion, step_id=step_id)
        result = self.midscene.assert_condition(
            assertion=assertion,
            report_path=report_path,
            step_log_file=self._shared_step_log_path(),
        ).payload
        self.run_context.logger.event("midscene_assert_done", step_id=step_id, result=result)
        return result

    def _shared_step_log_path(self) -> Path:
        return self.run_context.run_paths.logs_dir / "runlog.jsonl"

    def observe_current_state(
        self,
        *,
        round_idx: int,
        task: str,
        case_summary: str,
        label: str | None = None,
    ) -> StepObservation:
        try:
            qupath_state = self.query_qupath_state()
        except Exception as exc:
            qupath_state = {
                "ok": False,
                "error": f"Direct QuPath state query failed: {exc}",
            }
            self.run_context.logger.event("qupath_direct_state_error", round_idx=round_idx, error=qupath_state)

        qupath_state_summary = render_qupath_state_text(qupath_state, redact_sensitive=self.redact_model_state)
        gui_state = extract_rule_based_gui_state(qupath_state)
        gui_summary = render_qupath_gui_state_text(gui_state, redact_sensitive=self.redact_model_state)

        capture_label = label or f"round_{round_idx:04d}_view"
        viewer_capture = self.capture_viewer(label=capture_label)
        viewer_image_path = None
        if viewer_capture.get("ok") and viewer_capture.get("path"):
            viewer_image_path = Path(str(viewer_capture["path"]))

        observation = StepObservation(
            round_idx=round_idx,
            task=task,
            case_summary=case_summary,
            viewer_capture=viewer_capture,
            viewer_image_path=viewer_image_path,
            qupath_state=qupath_state,
            qupath_state_summary=qupath_state_summary,
            gui_state=gui_state,
            gui_summary=gui_summary,
        )
        self.run_context.logger.event(
            "qupath_observation",
            round_idx=round_idx,
            qupath_state=qupath_state,
            viewer_capture=viewer_capture,
            gui_state=gui_state,
        )
        return observation

    @staticmethod
    def _host_visible_direct_result(
        action_name: str,
        raw_payload: dict[str, Any],
        *,
        redact_sensitive: bool,
    ) -> dict[str, Any]:
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        text_output = QuPathConnector._direct_tool_text(action_name, payload, redact_sensitive=redact_sensitive)

        if not payload.get("ok"):
            result = {
                "ok": False,
                "action": action_name,
                "error": payload.get("error") or f"{action_name} failed.",
                "text_output": text_output,
            }
            diagnostics = {}
            if payload.get("stdout"):
                diagnostics["stdout"] = payload.get("stdout")
            if payload.get("stderr"):
                diagnostics["stderr"] = payload.get("stderr")
            if diagnostics:
                result["diagnostics"] = diagnostics
            return result

        if action_name == "zoom":
            return _host_zoom_result(payload, text_output)
        if action_name == "pan":
            return _host_pan_result(payload, text_output)
        if action_name == "viewer_point_to_wsi_coordinate":
            return _host_viewer_point_result(payload, text_output)
        if action_name == "set_image_type":
            return _host_image_type_result(payload, text_output)
        if action_name in {"open_project", "load_image"}:
            return _host_load_result(action_name, payload, text_output, redact_sensitive=redact_sensitive)

        return {
            "ok": True,
            "action": action_name,
            "text_output": text_output,
        }

    @staticmethod
    def _direct_tool_text(action_name: str, payload: dict[str, Any], *, redact_sensitive: bool) -> str:
        if not isinstance(payload, dict):
            return f"{action_name} returned a non-structured result."
        if not payload.get("ok"):
            error = payload.get("error")
            return f"{action_name} failed: {error}" if error else f"{action_name} failed."

        if action_name == "zoom":
            return _summarize_zoom_result(payload)
        if action_name == "pan":
            return _summarize_pan_result(payload)
        if action_name == "viewer_point_to_wsi_coordinate":
            return _summarize_viewer_point_result(payload)
        if action_name == "set_image_type":
            return _summarize_image_type_result(payload)
        if action_name in {"open_project", "load_image"}:
            return _summarize_load_result(action_name, payload, redact_sensitive=redact_sensitive)

        state = payload.get("state_after") or payload.get("state")
        state_text = render_qupath_state_text(
            {"ok": True, "state": state} if isinstance(state, dict) else {"ok": False},
            redact_sensitive=redact_sensitive,
        )
        return f"{action_name} succeeded.\n{state_text}".strip()

    def build_tool_handlers(self) -> dict[str, ToolHandler]:
        return {
            "load_case": lambda _args: self.load_case(),
            "open_project": lambda args: self.open_project(
                project_path=str(args.get("project_path", "")),
                image_name=str(args.get("image_name")) if args.get("image_name") is not None else None,
            ),
            "load_image": lambda args: self.load_image(image_path=str(args.get("image_path", ""))),
            "set_image_type": lambda args: self.set_image_type(image_type=str(args.get("image_type", ""))),
            "zoom": lambda args: self.zoom(
                direction=str(args.get("direction", "")),
                amount=float(args.get("amount", 1.0)),
                unit=str(args.get("unit", "")),
                x=_optional_float_arg(args.get("x")),
                y=_optional_float_arg(args.get("y")),
            ),
            "pan": lambda args: self.pan(
                direction=_optional_string_arg(args.get("direction")),
                amount=_optional_float_arg(args.get("amount")),
                unit=_optional_string_arg(args.get("unit")),
                mode=_optional_string_arg(args.get("mode")),
                x=_optional_float_arg(args.get("x")),
                y=_optional_float_arg(args.get("y")),
            ),
            "focus_qupath": lambda _args: self.focus_qupath(),
            "qupath_gui_act": lambda args: self.qupath_gui_act(instruction=str(args.get("instruction", ""))),
            "qupath_gui_query": lambda args: self.qupath_gui_query(question=str(args.get("question", ""))),
            "qupath_gui_assert": lambda args: self.qupath_gui_assert(assertion=str(args.get("assertion", ""))),
            "capture_viewer": lambda args: self.capture_viewer(label=str(args.get("label") or "viewer")),
            "viewer_point_to_wsi_coordinate": lambda args: self.viewer_point_to_wsi_coordinate(
                x=float(args.get("x", 0.0)),
                y=float(args.get("y", 0.0)),
            ),
            "record_current_view_as_evidence": lambda args: self.record_current_view_as_evidence(
                supported_label=str(args.get("supported_label", "")),
                description=str(args.get("description", "")),
            ),
            "sleep": lambda args: self.sleep(seconds=float(args.get("seconds", 1.0))),
            "close_qupath": lambda args: self.close_qupath(save=bool(args.get("save", False))),
        }



def _optional_float_arg(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_string_arg(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _viewer_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    viewer = state.get("viewer") or {}
    return viewer if isinstance(viewer, dict) else {}



def _summarize_zoom_result(payload: dict[str, Any]) -> str:
    before = _viewer_state(payload.get("state_before"))
    after = _viewer_state(payload.get("state_after") or payload.get("state"))
    parts = ["zoom succeeded."]

    unit = str(payload.get("unit") or "").strip()
    amount = payload.get("amount")
    target_mag = payload.get("target_magnification")
    effective_direction = str(payload.get("effective_direction") or "").strip()
    focus_mode = str(payload.get("focus_mode") or "").strip()

    if unit == "magnification":
        if target_mag is None:
            target_mag = amount
        parts.append(f"Target magnification: {_fmt_num(target_mag)}x.")
    elif unit == "relative_factor":
        parts.append(
            f"Relative zoom: {effective_direction or payload.get('direction') or '?'} by {_fmt_num(amount)}x from the current view."
        )
    elif unit == "steps":
        parts.append(
            f"Step zoom: {effective_direction or payload.get('direction') or '?'} by {_fmt_num(amount)} step(s)."
        )

    requested_viewer_x = payload.get("requested_viewer_x")
    requested_viewer_y = payload.get("requested_viewer_y")
    if focus_mode == "viewer_point" and requested_viewer_x is not None and requested_viewer_y is not None:
        parts.append(
            f"Recentered on normalized viewer point ({_fmt_num(requested_viewer_x)}, {_fmt_num(requested_viewer_y)})."
        )
    elif focus_mode == "current_viewer_center":
        parts.append("Used the current viewer center as the zoom focus.")

    before_mag = before.get("magnification")
    after_mag = after.get("magnification")
    before_downsample = before.get("downsample")
    after_downsample = after.get("downsample")

    if before_mag is not None and after_mag is not None:
        parts.append(f"Viewer magnification: {_fmt_num(before_mag)}x → {_fmt_num(after_mag)}x.")
    if before_downsample is not None and after_downsample is not None:
        parts.append(f"Viewer downsample: {_fmt_num(before_downsample)} → {_fmt_num(after_downsample)}.")

    before_center = _center_text(before)
    after_center = _center_text(after)
    if before_center and after_center:
        parts.append(f"Viewer center: {before_center} → {after_center}.")

    if payload.get("focus_was_clipped_to_image_bounds"):
        parts.append(
            "Warning: the requested viewer point was outside the image bounds and was clipped before recentering."
        )

    return " ".join(part.strip() for part in parts if part).strip()



def _summarize_pan_result(payload: dict[str, Any]) -> str:
    before = _viewer_state(payload.get("state_before"))
    after = _viewer_state(payload.get("state_after") or payload.get("state"))
    parts = ["pan succeeded."]
    mode = str(payload.get("mode") or "directional").strip()
    if mode == "set_center":
        requested_viewer_x = payload.get("requested_viewer_x")
        requested_viewer_y = payload.get("requested_viewer_y")
        if requested_viewer_x is not None and requested_viewer_y is not None:
            parts.append(
                f"Recentered on normalized viewer point ({_fmt_num(requested_viewer_x)}, {_fmt_num(requested_viewer_y)})."
            )
        else:
            parts.append("Recentered on the requested viewer point.")
        if payload.get("focus_was_clipped_to_image_bounds"):
            parts.append(
                "Warning: the requested viewer point was outside the image bounds and was clipped before recentering."
            )
    else:
        direction = str(payload.get("direction") or "").strip()
        amount = payload.get("amount")
        unit = str(payload.get("unit") or "").strip()
        if direction:
            parts.append(f"Direction: {direction}.")
        if amount is not None and unit:
            parts.append(f"Amount: {_fmt_num(amount)} {unit}.")

    before_center = _center_text(before)
    after_center = _center_text(after)
    if before_center and after_center:
        parts.append(f"Viewer center: {before_center} → {after_center}.")

    return " ".join(part.strip() for part in parts if part).strip()



def _summarize_viewer_point_result(payload: dict[str, Any]) -> str:
    viewer_x = payload.get("viewer_x")
    viewer_y = payload.get("viewer_y")
    image_x = payload.get("image_x")
    image_y = payload.get("image_y")
    parts = [
        "viewer_point_to_wsi_coordinate succeeded.",
        (
            f"Normalized viewer point ({_fmt_num(viewer_x)}, {_fmt_num(viewer_y)}) "
            f"maps to full-resolution image coordinate ({_fmt_num(image_x)}, {_fmt_num(image_y)})."
        ),
        "Coordinate system: top-left origin in full-resolution QuPath/WSI image pixels.",
    ]
    if payload.get("was_clipped_to_image_bounds"):
        parts.append(
            "Warning: the selected viewer point was outside the image bounds and was clipped. Pick a point on the main tissue image instead."
        )
    elif payload.get("inside_image_bounds") is False:
        parts.append(
            "Warning: the selected viewer point did not map cleanly inside the image bounds. Pick a point on the main tissue image instead."
        )
    return " ".join(part.strip() for part in parts if part).strip()



def _summarize_image_type_result(payload: dict[str, Any]) -> str:
    before_state = payload.get("state_before") or {}
    after_state = payload.get("state_after") or payload.get("state") or {}
    before_value = _image_type_text(before_state)
    after_value = _image_type_text(after_state)
    requested = payload.get("requested_image_type")
    if before_value and after_value and before_value != after_value:
        return f"set_image_type succeeded. Image type: {before_value} → {after_value}."
    if after_value:
        return f"set_image_type succeeded. Current image type: {after_value}. Requested: {requested}."
    return f"set_image_type succeeded. Requested image type: {requested}."



def _summarize_load_result(action_name: str, payload: dict[str, Any], *, redact_sensitive: bool) -> str:
    state = payload.get("state_after") or payload.get("state") or {}
    project_name = str(state.get("project_name") or "").strip() if isinstance(state, dict) else ""
    image_name = str(state.get("displayed_image_name") or "").strip() if isinstance(state, dict) else ""
    parts = [f"{action_name} succeeded."]
    if redact_sensitive:
        if project_name:
            parts.append("A project is loaded.")
        if image_name:
            parts.append("An image is displayed.")
    else:
        if project_name:
            parts.append(f"Project: {project_name}.")
        if image_name:
            parts.append(f"Displayed image: {image_name}.")
    return " ".join(parts).strip()



def _image_type_text(state: Any) -> str:
    if not isinstance(state, dict):
        return ""
    value = str(state.get("image_type_value") or "").strip()
    label = str(state.get("image_type_label") or "").strip()
    if value and label:
        return f"{value} ({label})"
    return value or label



def _center_tuple(viewer_state: dict[str, Any]) -> tuple[Any, Any] | None:
    x = viewer_state.get("center_pixel_x")
    y = viewer_state.get("center_pixel_y")
    if x is None or y is None:
        return None
    return (x, y)



def _center_text(viewer_state: dict[str, Any]) -> str:
    center = _center_tuple(viewer_state)
    if center is None:
        return ""
    x, y = center
    return f"({_fmt_num(x)}, {_fmt_num(y)})"



def _fmt_num(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:.3f}"
    return str(value)


def _host_load_result(action_name: str, payload: dict[str, Any], text_output: str, *, redact_sensitive: bool) -> dict[str, Any]:
    state = payload.get("state_after") or payload.get("state") or {}
    result = {
        "ok": True,
        "action": action_name,
        "text_output": text_output,
    }
    if action_name == "open_project":
        requested_project_path = payload.get("requested_project_path") or payload.get("project_path")
        requested_project_image_name = payload.get("requested_project_image_name") or payload.get("image_name")
        if requested_project_path is not None and not redact_sensitive:
            result["requested_project_path"] = requested_project_path
        if requested_project_image_name is not None and not redact_sensitive:
            result["requested_project_image_name"] = requested_project_image_name
    if action_name == "load_image":
        requested_image_path = payload.get("requested_image_path") or payload.get("image_path")
        if requested_image_path is not None and not redact_sensitive:
            result["requested_image_path"] = requested_image_path
    if isinstance(state, dict):
        project_name = state.get("project_name")
        displayed_image_name = state.get("displayed_image_name")
        project_path = state.get("project_path")
        if project_name and not redact_sensitive:
            result["project_name"] = project_name
        if project_path and not redact_sensitive:
            result["project_path"] = project_path
        if displayed_image_name and not redact_sensitive:
            result["displayed_image_name"] = displayed_image_name
        if redact_sensitive:
            result["project_loaded"] = bool(project_name)
            result["image_loaded"] = bool(displayed_image_name)
    if payload.get("mode"):
        result["mode"] = payload.get("mode")
    return result


def _host_image_type_result(payload: dict[str, Any], text_output: str) -> dict[str, Any]:
    before_state = payload.get("state_before") or {}
    after_state = payload.get("state_after") or payload.get("state") or {}
    result = {
        "ok": True,
        "action": "set_image_type",
        "requested_image_type": payload.get("requested_image_type"),
        "text_output": text_output,
    }
    previous_type = _image_type_text(before_state)
    current_type = _image_type_text(after_state)
    if previous_type:
        result["previous_image_type"] = previous_type
    if current_type:
        result["current_image_type"] = current_type
    if payload.get("supported_image_types"):
        result["supported_image_types"] = payload.get("supported_image_types")
    return result


def _host_zoom_result(payload: dict[str, Any], text_output: str) -> dict[str, Any]:
    before = _viewer_state(payload.get("state_before"))
    after = _viewer_state(payload.get("state_after") or payload.get("state"))
    result = {
        "ok": True,
        "action": "zoom",
        "unit": payload.get("unit"),
        "requested_direction": payload.get("direction"),
        "requested_amount": payload.get("amount"),
        "effective_direction": payload.get("effective_direction"),
        "focus_mode": payload.get("focus_mode") or "current_viewer_center",
        "text_output": text_output,
    }
    if payload.get("target_magnification") is not None:
        result["target_magnification"] = payload.get("target_magnification")
    for key in (
        "before_magnification",
        "after_magnification",
        "before_downsample",
        "after_downsample",
    ):
        if payload.get(key) is not None:
            result[key] = payload.get(key)
    if payload.get("requested_viewer_x") is not None and payload.get("requested_viewer_y") is not None:
        result["requested_viewer_point"] = {
            "x": payload.get("requested_viewer_x"),
            "y": payload.get("requested_viewer_y"),
        }
    if payload.get("focus_image_x") is not None and payload.get("focus_image_y") is not None:
        result["focus_image_point"] = {
            "x": payload.get("focus_image_x"),
            "y": payload.get("focus_image_y"),
        }
    if payload.get("focus_was_clipped_to_image_bounds") is not None:
        result["focus_was_clipped_to_image_bounds"] = payload.get("focus_was_clipped_to_image_bounds")
    before_center = _center_tuple(before)
    after_center = _center_tuple(after)
    if before_center is not None:
        result["center_before"] = {"x": before_center[0], "y": before_center[1]}
    if after_center is not None:
        result["center_after"] = {"x": after_center[0], "y": after_center[1]}
    return result


def _host_pan_result(payload: dict[str, Any], text_output: str) -> dict[str, Any]:
    before = _viewer_state(payload.get("state_before"))
    after = _viewer_state(payload.get("state_after") or payload.get("state"))
    result = {
        "ok": True,
        "action": "pan",
        "mode": payload.get("mode") or "directional",
        "direction": payload.get("direction"),
        "amount": payload.get("amount"),
        "unit": payload.get("unit"),
        "text_output": text_output,
    }
    if payload.get("delta_x") is not None:
        result["delta_x"] = payload.get("delta_x")
    if payload.get("delta_y") is not None:
        result["delta_y"] = payload.get("delta_y")
    if payload.get("requested_viewer_x") is not None and payload.get("requested_viewer_y") is not None:
        result["requested_viewer_point"] = {
            "x": payload.get("requested_viewer_x"),
            "y": payload.get("requested_viewer_y"),
        }
    if payload.get("focus_image_x") is not None and payload.get("focus_image_y") is not None:
        result["focus_image_point"] = {
            "x": payload.get("focus_image_x"),
            "y": payload.get("focus_image_y"),
        }
    if payload.get("focus_was_clipped_to_image_bounds") is not None:
        result["focus_was_clipped_to_image_bounds"] = payload.get("focus_was_clipped_to_image_bounds")
    before_center = _center_tuple(before)
    after_center = _center_tuple(after)
    if before_center is not None:
        result["center_before"] = {"x": before_center[0], "y": before_center[1]}
    if after_center is not None:
        result["center_after"] = {"x": after_center[0], "y": after_center[1]}
    return result



def _host_viewer_point_result(payload: dict[str, Any], text_output: str) -> dict[str, Any]:
    result = {
        "ok": True,
        "action": "viewer_point_to_wsi_coordinate",
        "viewer_x": payload.get("viewer_x"),
        "viewer_y": payload.get("viewer_y"),
        "x": payload.get("image_x"),
        "y": payload.get("image_y"),
        "coordinate_system": payload.get("coordinate_system") or "qupath_full_resolution_image_pixels",
        "inside_image_bounds": payload.get("inside_image_bounds"),
        "was_clipped_to_image_bounds": payload.get("was_clipped_to_image_bounds"),
        "text_output": text_output,
    }
    for key in (
        "raw_image_x",
        "raw_image_y",
        "component_x",
        "component_y",
        "viewer_width",
        "viewer_height",
        "image_width",
        "image_height",
    ):
        if payload.get(key) is not None:
            result[key] = payload.get(key)
    return result
