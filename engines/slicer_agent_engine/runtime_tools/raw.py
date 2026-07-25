
from __future__ import annotations

import difflib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

from .common import _as_path, _canonical_view, _normalize_public_view_payload, format_scene_inventory_for_prompt, logger

if TYPE_CHECKING:
    from ..algorithm_runtime import AlgorithmRuntime
    from ..gemini_video import GeminiVideoAnalyzer

class ToolContextRawMixin:
    def set_gui(self, *, contents: Optional[str] = None, viewers_layout: Optional[str] = None) -> Dict[str, Any]:
        args = {"contents": contents, "viewers_layout": viewers_layout}
        try:
            result = self.client.put_gui(contents=contents, viewers_layout=viewers_layout)
            self._log(tool="set_gui", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_gui", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # MRML endpoints (raw WebServer)
    # -------------------------

    def mrml_list_nodes(self, *, classname: Optional[str] = None) -> Any:
        args = {"classname": classname}
        try:
            result = self.client.mrml_list_nodes(classname=classname)
            self._log(tool="mrml_list_nodes", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_list_nodes", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_names(self, *, classname: Optional[str] = None) -> Any:
        args = {"classname": classname}
        try:
            result = self.client.mrml_names(classname=classname)
            self._log(tool="mrml_names", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_names", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_ids(self, *, classname: Optional[str] = None) -> Any:
        args = {"classname": classname}
        try:
            result = self.client.mrml_ids(classname=classname)
            self._log(tool="mrml_ids", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_ids", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_properties(self, *, node_id: str) -> Any:
        args = {"node_id": node_id}
        try:
            result = self.client.mrml_properties(node_id=node_id)
            self._log(tool="mrml_properties", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_properties", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_set_properties(self, *, node_id: str, properties: Dict[str, Any]) -> Any:
        args = {"node_id": node_id, "properties": properties}
        try:
            result = self.client.mrml_set_properties(node_id=node_id, properties=properties)
            self._log(tool="mrml_set_properties", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_set_properties", args=args, result=None, ok=False, error=str(e))
            raise


    def mrml_file_get(self, *, node_id: str, file_name: str) -> Any:
        args = {"node_id": node_id, "file_name": file_name}
        try:
            result = self.client.mrml_file_get(node_id=node_id, file_name=file_name)
            self._log(tool="mrml_file_get", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_file_get", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_file_post(self, *, file_name: str) -> Any:
        args = {"file_name": file_name}
        try:
            result = self.client.mrml_file_post(file_name=file_name)
            self._log(tool="mrml_file_post", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_file_post", args=args, result=None, ok=False, error=str(e))
            raise

    def mrml_file_delete(self, *, node_id: Optional[str] = None, file_name: Optional[str] = None) -> Any:
        args = {"node_id": node_id, "file_name": file_name}
        try:
            result = self.client.mrml_file_delete(node_id=node_id, file_name=file_name)
            self._log(tool="mrml_file_delete", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="mrml_file_delete", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Volume selection / transforms
    # -------------------------

    def volume_selection(self, *, cmd: str) -> Any:
        args = {"cmd": cmd}
        try:
            result = self.client.volume_selection(cmd=cmd)
            self._log(tool="volume_selection", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="volume_selection", args=args, result=None, ok=False, error=str(e))
            raise

    def list_gridtransforms(self) -> Any:
        args: Dict[str, Any] = {}
        try:
            result = self.client.list_gridtransforms()
            self._log(tool="list_gridtransforms", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_gridtransforms", args=args, result=None, ok=False, error=str(e))
            raise

    def download_volume_nrrd(self, *, node_id: str, out_name: Optional[str] = None) -> Dict[str, Any]:
        args = {"node_id": node_id, "out_name": out_name}
        try:
            if out_name is None:
                out_name = f"volume_{node_id}.nrrd"
            out_path = self._artifact_path(out_name)
            self.client.download_volume_nrrd(node_id=node_id, out_path=out_path)
            result = {"nrrd_path": str(out_path)}
            self._log(tool="download_volume_nrrd", args=args, result=result, artifacts={"nrrd": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="download_volume_nrrd", args=args, result=None, ok=False, error=str(e))
            raise

    def upload_volume_nrrd(self, *, nrrd_path: Union[str, Path]) -> Any:
        nrrd_path_p = _as_path(nrrd_path)
        args = {"nrrd_path": str(nrrd_path_p)}
        try:
            result = self.client.upload_volume_nrrd(nrrd_path=nrrd_path_p)
            self._log(tool="upload_volume_nrrd", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="upload_volume_nrrd", args=args, result=None, ok=False, error=str(e))
            raise

    def download_gridtransform_nrrd(self, *, node_id: str, out_name: Optional[str] = None) -> Dict[str, Any]:
        args = {"node_id": node_id, "out_name": out_name}
        try:
            if out_name is None:
                out_name = f"gridtransform_{node_id}.nrrd"
            out_path = self._artifact_path(out_name)
            self.client.download_gridtransform_nrrd(node_id=node_id, out_path=out_path)
            result = {"nrrd_path": str(out_path)}
            self._log(tool="download_gridtransform_nrrd", args=args, result=result, artifacts={"nrrd": str(out_path)})
            return result
        except Exception as e:
            self._log(tool="download_gridtransform_nrrd", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Fiducials
    # -------------------------

    def list_fiducials(self) -> Any:
        args: Dict[str, Any] = {}
        try:
            result = self.client.list_fiducials()
            self._log(tool="list_fiducials", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="list_fiducials", args=args, result=None, ok=False, error=str(e))
            raise

    def set_fiducial_ras(self, *, node_id: str, r: float, a: float, s: float) -> Any:
        args = {"node_id": node_id, "r": r, "a": a, "s": s}
        try:
            result = self.client.put_fiducial(node_id=node_id, r=r, a=a, s=s)
            self._log(tool="set_fiducial_ras", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="set_fiducial_ras", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Tracking
    # -------------------------

    def get_tracking(self, *, m: Optional[str] = None, q: Optional[str] = None, p: Optional[str] = None) -> Any:
        args = {"m": m, "q": q, "p": p}
        try:
            result = self.client.get_tracking(m=m, q=q, p=p)
            self._log(tool="get_tracking", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="get_tracking", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Sample data
    # -------------------------

    def load_sampledata(self, *, name: str) -> Any:
        args = {"name": name}
        try:
            result = self.client.load_sampledata(name=name)
            self._log(tool="load_sampledata", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="load_sampledata", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # DICOMweb (Slicer endpoint)
    # -------------------------

    def access_dicomweb_study(self, payload: Dict[str, Any]) -> Any:
        args = {"payload": payload}
        try:
            result = self.client.access_dicomweb_study(payload)
            self._log(tool="access_dicomweb_study", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="access_dicomweb_study", args=args, result=None, ok=False, error=str(e))
            raise

    # -------------------------
    # Raw requests for completeness
    # -------------------------

    def raw_slicer_request(
        self,
        *,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data_text: Optional[str] = None,
        save_as: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generic HTTP call to the `/slicer` subtree.

        Use this to access newly-added WebServer endpoints without changing this repo.
        """
        args = {"method": method, "path": path, "params": params, "json_body": json_body, "save_as": save_as}
        try:
            data: Optional[bytes] = None
            if data_text is not None:
                data = data_text.encode("utf-8")
            resp = self.client.raw_request(
                method,
                path,
                params=params,
                json_body=json_body,
                data=data,
                stream=bool(save_as),
            )
            headers = dict(resp.headers)
            ct = headers.get("Content-Type", "")

            if save_as:
                out_path = self._artifact_path(save_as)
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                result = {
                    "status_code": resp.status_code,
                    "content_type": ct,
                    "headers": headers,
                    "saved_path": str(out_path),
                }
                self._log(tool="raw_slicer_request", args=args, result=result, artifacts={"file": str(out_path)})
                return result

            # Try JSON first
            try:
                payload = resp.json()
                result = {"status_code": resp.status_code, "content_type": ct, "headers": headers, "json": payload}
            except Exception:
                result = {"status_code": resp.status_code, "content_type": ct, "headers": headers, "text": resp.text}

            self._log(tool="raw_slicer_request", args=args, result=result)
            return result

        except Exception as e:
            self._log(tool="raw_slicer_request", args=args, result=None, ok=False, error=str(e))
            raise

    def raw_dicomweb_request(
        self,
        *,
        method: str,
        dicom_path: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        data_text: Optional[str] = None,
        save_as: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generic HTTP call to the `/dicom` subtree (DICOMweb).

        Slicer can expose its internal DICOM database as DICOMweb.
        If you did not enable DICOMweb in WebServer, this will 404.
        """
        args = {
            "method": method,
            "dicom_path": dicom_path,
            "params": params,
            "headers": headers,
            "save_as": save_as,
        }
        try:
            data: Optional[bytes] = None
            if data_text is not None:
                data = data_text.encode("utf-8")
            resp = self.client.dicomweb_request(
                method,
                dicom_path,
                params=params,
                headers=headers,
                data=data,
                stream=bool(save_as),
            )
            resp_headers = dict(resp.headers)
            ct = resp_headers.get("Content-Type", "")

            if save_as:
                out_path = self._artifact_path(save_as)
                with out_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                result = {
                    "status_code": resp.status_code,
                    "content_type": ct,
                    "headers": resp_headers,
                    "saved_path": str(out_path),
                }
                self._log(tool="raw_dicomweb_request", args=args, result=result, artifacts={"file": str(out_path)})
                return result

            try:
                payload = resp.json()
                result = {"status_code": resp.status_code, "content_type": ct, "headers": resp_headers, "json": payload}
            except Exception:
                result = {"status_code": resp.status_code, "content_type": ct, "headers": resp_headers, "text": resp.text}

            self._log(tool="raw_dicomweb_request", args=args, result=result)
            return result

        except Exception as e:
            self._log(tool="raw_dicomweb_request", args=args, result=None, ok=False, error=str(e))
            raise



    def exec_python(self, *, source: str) -> Dict[str, Any]:
        """Run raw Python in Slicer via `/slicer/exec`.

        **Unsafe**: do not expose to untrusted callers. Prefer the fixed-entry bridge tools.
        """
        args = {"source": source}
        try:
            result = self.client.exec_python(source)
            self._log(tool="exec_python", args=args, result=result)
            return result
        except Exception as e:
            self._log(tool="exec_python", args=args, result=None, ok=False, error=str(e))
            raise

    def exec_bridge(self, *, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call any bridge tool by name.

        This is a forward-compatibility hook: you can add new bridge tools on the Slicer side
        without updating the MCP schema immediately.
        """
        call_args = {"tool": tool, "args": args}
        try:
            result = self.client.exec_bridge(bridge_dir=self.bridge_dir, tool=tool, args=args, session_id=self.session.session_id)
            self._log(tool="exec_bridge", args=call_args, result=result)
            return result
        except Exception as e:
            self._log(tool="exec_bridge", args=call_args, result=None, ok=False, error=str(e))
            raise
