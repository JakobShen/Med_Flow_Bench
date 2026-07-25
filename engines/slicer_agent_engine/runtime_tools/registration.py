from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class ToolContextRegistrationMixin:
    def _apply_registration_state_from_result(self, result: Dict[str, Any]) -> None:
        updates: Dict[str, Any] = {}
        if result.get("transform_id") is not None:
            updates["active_transform_id"] = result.get("transform_id")
        if result.get("transform_name") is not None:
            updates["active_transform_name"] = result.get("transform_name")
        if result.get("fixed_volume_id") is not None:
            updates["last_registration_fixed_volume_id"] = result.get("fixed_volume_id")
        if result.get("fixed_volume_name") is not None:
            updates["last_registration_fixed_volume_name"] = result.get("fixed_volume_name")
        if result.get("moving_volume_id") is not None:
            updates["last_registration_moving_volume_id"] = result.get("moving_volume_id")
        if result.get("moving_volume_name") is not None:
            updates["last_registration_moving_volume_name"] = result.get("moving_volume_name")
        if result.get("output_volume_id") is not None:
            updates["last_registration_output_volume_id"] = result.get("output_volume_id")
        if result.get("output_volume_name") is not None:
            updates["last_registration_output_volume_name"] = result.get("output_volume_name")
        if result.get("sampled_transform_id") is not None:
            updates["active_transform_id"] = result.get("sampled_transform_id")
        if result.get("sampled_transform_name") is not None:
            updates["active_transform_name"] = result.get("sampled_transform_name")
        if result.get("sampled_volume_id") is not None and result.get("made_active"):
            updates["active_volume_id"] = result.get("sampled_volume_id")
        if result.get("sampled_volume_name") is not None and result.get("made_active"):
            updates["active_volume_name"] = result.get("sampled_volume_name")
        if result.get("output_volume_id") is not None and result.get("made_active"):
            updates["active_volume_id"] = result.get("output_volume_id")
        if result.get("output_volume_name") is not None and result.get("made_active"):
            updates["active_volume_name"] = result.get("output_volume_name")
        if result.get("active_volume_id") is not None:
            updates["active_volume_id"] = result.get("active_volume_id")
        if result.get("active_volume_name") is not None:
            updates["active_volume_name"] = result.get("active_volume_name")
        if updates:
            self.session.update_state(**updates)

    def _registration_bridge_call(self, *, tool: str, args: Dict[str, Any], timeout_s: Optional[float] = None) -> Dict[str, Any]:
        result = self.client.exec_bridge(
            bridge_dir=self.bridge_dir,
            tool=tool,
            args=args,
            session_id=self.session.session_id,
            timeout_s=timeout_s,
        )
        if isinstance(result, dict):
            self._apply_registration_state_from_result(result)
        return result

    def _transform_arg_bundle(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "transform": transform,
            "transform_id": transform_id or self.session.state.get("active_transform_id"),
            "transform_name": transform_name or self.session.state.get("active_transform_name"),
        }

    def _node_arg_bundle(
        self,
        *,
        node_refs: Optional[Sequence[str]] = None,
        node_ids: Optional[Sequence[str]] = None,
        node_names: Optional[Sequence[str]] = None,
        node_ref: Optional[str] = None,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "node_refs": list(node_refs) if node_refs is not None else None,
            "node_ids": list(node_ids) if node_ids is not None else None,
            "node_names": list(node_names) if node_names is not None else None,
            "node_ref": node_ref,
            "node_id": node_id,
            "node_name": node_name,
        }

    def list_transforms(self) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        try:
            result = self._registration_bridge_call(tool="list_transforms", args=args)
            self._log(tool="list_transforms", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_transforms", args=args, result=None, ok=False, error=str(e))
            raise

    def create_transform(self, *, name: Optional[str] = None, kind: str = "general", make_active: bool = True) -> Dict[str, Any]:
        args = {"name": name, "kind": kind, "make_active": bool(make_active)}
        try:
            result = self._registration_bridge_call(tool="create_transform", args=args)
            self._log(tool="create_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="create_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def get_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name)
        try:
            result = self._registration_bridge_call(tool="get_transform", args=args)
            self._log(tool="get_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="get_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def edit_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
        matrix_4x4: Optional[Sequence[Sequence[float]]] = None,
        translation: Optional[Sequence[float]] = None,
        rotation_degrees: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        shear: Optional[Sequence[float]] = None,
        translation_reference_frame: str = "parent",
        reset_to_identity: bool = False,
    ) -> Dict[str, Any]:
        args = {
            **self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name),
            "matrix_4x4": [list(row) for row in matrix_4x4] if matrix_4x4 is not None else None,
            "translation": [float(v) for v in translation] if translation is not None else None,
            "rotation_degrees": [float(v) for v in rotation_degrees] if rotation_degrees is not None else None,
            "scale": [float(v) for v in scale] if scale is not None else None,
            "shear": [float(v) for v in shear] if shear is not None else None,
            "translation_reference_frame": str(translation_reference_frame),
            "reset_to_identity": bool(reset_to_identity),
        }
        try:
            result = self._registration_bridge_call(tool="edit_transform", args=args)
            self._log(tool="edit_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="edit_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def set_transform_center(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
        mode: str = "world_ras",
        ras: Optional[Sequence[float]] = None,
        local: Optional[Sequence[float]] = None,
        node_ref: Optional[str] = None,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            **self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name),
            "mode": str(mode),
            "ras": [float(v) for v in ras] if ras is not None else None,
            "local": [float(v) for v in local] if local is not None else None,
            "node_ref": node_ref,
            "node_id": node_id,
            "node_name": node_name,
        }
        try:
            result = self._registration_bridge_call(tool="set_transform_center", args=args)
            self._log(tool="set_transform_center", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_transform_center", args=args, result=None, ok=False, error=str(e))
            raise

    def apply_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
        node_refs: Optional[Sequence[str]] = None,
        node_ids: Optional[Sequence[str]] = None,
        node_names: Optional[Sequence[str]] = None,
        node_ref: Optional[str] = None,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            **self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name),
            **self._node_arg_bundle(
                node_refs=node_refs,
                node_ids=node_ids,
                node_names=node_names,
                node_ref=node_ref,
                node_id=node_id,
                node_name=node_name,
            ),
        }
        try:
            result = self._registration_bridge_call(tool="apply_transform", args=args)
            self._log(tool="apply_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="apply_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def remove_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
        node_refs: Optional[Sequence[str]] = None,
        node_ids: Optional[Sequence[str]] = None,
        node_names: Optional[Sequence[str]] = None,
        node_ref: Optional[str] = None,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            **self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name),
            **self._node_arg_bundle(
                node_refs=node_refs,
                node_ids=node_ids,
                node_names=node_names,
                node_ref=node_ref,
                node_id=node_id,
                node_name=node_name,
            ),
        }
        try:
            result = self._registration_bridge_call(tool="remove_transform", args=args)
            self._log(tool="remove_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="remove_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def harden_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
        node_refs: Optional[Sequence[str]] = None,
        node_ids: Optional[Sequence[str]] = None,
        node_names: Optional[Sequence[str]] = None,
        node_ref: Optional[str] = None,
        node_id: Optional[str] = None,
        node_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            **self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name),
            **self._node_arg_bundle(
                node_refs=node_refs,
                node_ids=node_ids,
                node_names=node_names,
                node_ref=node_ref,
                node_id=node_id,
                node_name=node_name,
            ),
        }
        try:
            result = self._registration_bridge_call(tool="harden_transform", args=args)
            self._log(tool="harden_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="harden_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def invert_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name)
        try:
            result = self._registration_bridge_call(tool="invert_transform", args=args)
            self._log(tool="invert_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="invert_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def compose_transforms(
        self,
        *,
        child_transform: Optional[str],
        child_transform_id: Optional[str] = None,
        child_transform_name: Optional[str] = None,
        parent_transform: Optional[str],
        parent_transform_id: Optional[str] = None,
        parent_transform_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = {
            "child_transform": child_transform,
            "child_transform_id": child_transform_id,
            "child_transform_name": child_transform_name,
            "parent_transform": parent_transform,
            "parent_transform_id": parent_transform_id,
            "parent_transform_name": parent_transform_name,
        }
        try:
            result = self._registration_bridge_call(tool="compose_transforms", args=args)
            self._log(tool="compose_transforms", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="compose_transforms", args=args, result=None, ok=False, error=str(e))
            raise

    def split_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        args = self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name)
        try:
            result = self._registration_bridge_call(tool="split_transform", args=args)
            self._log(tool="split_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="split_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def sample_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
        reference: Optional[str] = None,
        reference_volume: Optional[str] = None,
        reference_volume_id: Optional[str] = None,
        reference_volume_name: Optional[str] = None,
        output_kind: str = "grid_transform",
        output_name: Optional[str] = None,
        make_active: bool = False,
    ) -> Dict[str, Any]:
        args = {
            **self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name),
            "reference": reference,
            "referenceVolume": reference_volume,
            "reference_volume_id": reference_volume_id,
            "reference_volume_name": reference_volume_name,
            "output_kind": output_kind,
            "output_name": output_name,
            "make_active": bool(make_active),
        }
        try:
            result = self._registration_bridge_call(tool="sample_transform", args=args)
            self._log(tool="sample_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="sample_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def save_transform(
        self,
        *,
        transform: Optional[str] = None,
        transform_id: Optional[str] = None,
        transform_name: Optional[str] = None,
        output_path: str | Path,
    ) -> Dict[str, Any]:
        args = {
            **self._transform_arg_bundle(transform=transform, transform_id=transform_id, transform_name=transform_name),
            "output_path": str(Path(output_path).expanduser()),
        }
        try:
            result = self._registration_bridge_call(tool="save_transform", args=args)
            self._log(tool="save_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="save_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def load_transform(self, *, path: str | Path, make_active: bool = True) -> Dict[str, Any]:
        args = {"path": str(Path(path).expanduser()), "make_active": bool(make_active)}
        try:
            result = self._registration_bridge_call(tool="load_transform", args=args)
            self._log(tool="load_transform", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_transform", args=args, result=None, ok=False, error=str(e))
            raise

    def brains_fit(self, **kwargs: Any) -> Dict[str, Any]:
        args = dict(kwargs)
        if not args.get("output") and not args.get("output_transform") and not args.get("linearTransform") and not args.get("bsplineTransform"):
            args["output"] = "BRAINSFit Transform"
        timeout_s = float(max(self.client.timeout_s, 600.0))
        try:
            result = self._registration_bridge_call(tool="brains_fit", args=args, timeout_s=timeout_s)
            self._log(tool="brains_fit", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="brains_fit", args=args, result=None, ok=False, error=str(e))
            raise

    def brains_resample(self, **kwargs: Any) -> Dict[str, Any]:
        args = dict(kwargs)
        if args.get("transform") is None and self.session.state.get("active_transform_id"):
            args["transform"] = self.session.state.get("active_transform_id")
        if args.get("output") is None and args.get("outputVolume") is None:
            args["output"] = "BRAINSResample Output"
        timeout_s = float(max(self.client.timeout_s, 300.0))
        try:
            result = self._registration_bridge_call(tool="brains_resample", args=args, timeout_s=timeout_s)
            self._log(tool="brains_resample", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="brains_resample", args=args, result=None, ok=False, error=str(e))
            raise

    def brains_roi_auto(self, **kwargs: Any) -> Dict[str, Any]:
        args = dict(kwargs)
        timeout_s = float(max(self.client.timeout_s, 300.0))
        try:
            result = self._registration_bridge_call(tool="brains_roi_auto", args=args, timeout_s=timeout_s)
            self._log(tool="brains_roi_auto", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="brains_roi_auto", args=args, result=None, ok=False, error=str(e))
            raise
