#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _load_request(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_response(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_volume_for_m3d(nifti_path: Path, *, target_shape: Tuple[int, int, int]) -> np.ndarray:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError("nibabel is required for the M3D worker.") from e
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("torch is required for the M3D worker.") from e

    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {nifti_path}, got shape {list(vol.shape)}")

    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float32)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    vol = np.clip(vol, 0.0, 1.0).astype(np.float32, copy=False)

    # M3D quickstart expects 1 x 32 x 256 x 256 before adding batch dim.
    vol_dhw = np.transpose(vol, (2, 1, 0))
    tensor = torch.from_numpy(vol_dhw).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=tuple(int(v) for v in target_shape),
        mode="trilinear",
        align_corners=False,
    )
    arr = tensor[0, 0].unsqueeze(0).cpu().numpy().astype(np.float32, copy=False)
    return arr


def _resolve_torch_dtype(dtype_name: str):
    import torch

    key = str(dtype_name or "bfloat16").strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported M3D dtype: {dtype_name!r}")
    return mapping[key]


def run_m3d(request: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("The M3D worker requires torch and transformers.") from e

    nifti_path = Path(str(request["nifti_path"])).expanduser().resolve()
    if not nifti_path.exists():
        raise FileNotFoundError(f"Input NIfTI volume does not exist: {nifti_path}")

    model_name_or_path = str(request.get("model_name_or_path") or "GoodBaiBai88/M3D-LaMed-Phi-3-4B")
    question = str(request.get("question") or "")
    proj_out_num = int(request.get("proj_out_num") or 256)
    max_new_tokens = int(request.get("max_new_tokens") or 256)
    device = str(request.get("device") or "cuda")
    dtype_name = str(request.get("dtype") or "bfloat16")
    target_shape = tuple(int(v) for v in (request.get("target_shape") or [32, 256, 256]))
    if len(target_shape) != 3:
        raise ValueError(f"Expected target_shape of length 3, got {target_shape}")

    torch_dtype = _resolve_torch_dtype(dtype_name)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("M3D worker requested CUDA but torch.cuda.is_available() is False.")

    image_np = _normalize_volume_for_m3d(nifti_path, target_shape=target_shape)
    image_pt = torch.from_numpy(image_np).unsqueeze(0)
    image_pt = image_pt.to(device=device, dtype=torch_dtype)

    load_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if device.startswith("cuda"):
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    if not device.startswith("cuda"):
        model = model.to(device=device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=int(request.get("model_max_length") or 512),
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device=device)

    with torch.no_grad():
        generation = model.generate(
            image_pt,
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=bool(request.get("do_sample", False)),
            top_p=float(request.get("top_p") or 0.9),
            temperature=float(request.get("temperature") or 1.0),
        )
    if isinstance(generation, tuple):
        generation = generation[0]
    generated_text = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]

    return {
        "ok": True,
        "worker": "m3d_worker",
        "nifti_path": str(nifti_path),
        "model_name_or_path": model_name_or_path,
        "device": device,
        "dtype": dtype_name,
        "target_shape": [int(v) for v in target_shape],
        "question": question,
        "text": str(generated_text or ""),
        "preprocess": {
            "normalized_shape": [int(v) for v in image_np.shape],
            "normalized_min": float(image_np.min()),
            "normalized_max": float(image_np.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track C M3D worker")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    req_path = Path(args.input_json).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()
    request = _load_request(req_path)
    try:
        payload = run_m3d(request)
    except Exception as e:
        payload = {
            "ok": False,
            "worker": "m3d_worker",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        _write_response(out_path, payload)
        raise
    _write_response(out_path, payload)


if __name__ == "__main__":
    main()
