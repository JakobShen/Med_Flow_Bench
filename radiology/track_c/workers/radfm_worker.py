#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


_IMAGE_TOKEN_RE = re.compile(r"</?image\d*>", re.IGNORECASE)
_EVIDENCE_KEYS = (
    "KEY_SLICE_AXIAL_MM:",
    "KEY_SLICE_SAGITTAL_MM:",
    "KEY_SLICE_CORONAL_MM:",
    "RSA:",
)


def _load_request(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_response(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _normalize_valid_choices(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [part.strip().upper() for part in raw.split(",")]
    elif isinstance(raw, Sequence):
        items = [str(part).strip().upper() for part in raw]
    else:
        items = [str(raw).strip().upper()]
    out: List[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return tuple(out)


def _import_radfm_model(repo_root: Path):
    candidate_sys_paths = [repo_root / "Quick_demo", repo_root / "src", repo_root]
    for candidate in candidate_sys_paths:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import RadFM model code. Check that --radfm_repo_root points to a cloned RadFM repo with either Quick_demo/Model or src/Model available."
        ) from e
    return MultiLLaMAForCausalLM


def _patch_llama_from_pretrained(torch, device: str) -> None:
    from transformers.models.llama import LlamaForCausalLM

    if getattr(LlamaForCausalLM, "_radfm_track_c_patched", False):
        return

    original = getattr(LlamaForCausalLM.from_pretrained, "__func__", None)
    if original is None:
        return

    def wrapped(cls, pretrained_model_name_or_path, *args, **kwargs):
        kwargs.setdefault("torch_dtype", torch.float16)
        kwargs.setdefault("low_cpu_mem_usage", True)
        kwargs.setdefault("device_map", {"": device})
        return original(cls, pretrained_model_name_or_path, *args, **kwargs)

    LlamaForCausalLM.from_pretrained = classmethod(wrapped)
    setattr(LlamaForCausalLM, "_radfm_track_c_patched", True)


def _resolve_language_dir(repo_root: Path, language_dir: Optional[str]) -> Path:
    if language_dir:
        raw = str(language_dir).strip()
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
    candidates = [
        repo_root / "Quick_demo" / "Language_files",
        repo_root / "Language_files",
        repo_root / "src" / "Language_files",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not locate RadFM Language_files under {repo_root}. Pass --radfm_language_dir explicitly."
    )


def _get_tokenizer(tokenizer_path: Path, *, max_img_size: int = 10, image_num: int = 32):
    try:
        from transformers import LlamaTokenizer
    except Exception as e:
        raise RuntimeError("transformers is required for the RadFM worker.") from e

    text_tokenizer = LlamaTokenizer.from_pretrained(str(tokenizer_path))
    special_tokens: Dict[str, List[str]] = {"additional_special_tokens": ["<image>", "</image>"]}
    image_padding_tokens: List[str] = []
    for img_idx in range(max_img_size):
        image_padding_token = ""
        for patch_idx in range(image_num):
            image_token = f"<image{img_idx * image_num + patch_idx}>"
            image_padding_token += image_token
            special_tokens["additional_special_tokens"].append(image_token)
        image_padding_tokens.append(image_padding_token)
    text_tokenizer.add_special_tokens(special_tokens)
    text_tokenizer.pad_token_id = 0
    text_tokenizer.bos_token_id = 1
    text_tokenizer.eos_token_id = 2
    return text_tokenizer, image_padding_tokens


def _nearest_supported_depth(depth: int) -> int:
    allowed = list(range(4, 65, 4))
    if depth <= 0:
        return 32
    return min(allowed, key=lambda v: abs(v - int(depth)))


def _load_nifti_volume(nifti_path: Path):
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError("nibabel is required for the RadFM worker.") from e

    img = nib.load(str(nifti_path))
    vol = np.asarray(img.get_fdata(dtype=np.float32))
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI volume at {nifti_path}, got shape {list(vol.shape)}")
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    return img, vol.astype(np.float32, copy=False)


def _robust_normalize(vol: np.ndarray) -> np.ndarray:
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float32)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip(vol, lo, hi)
    out = (out - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _normalize_volume_for_radfm(nifti_path: Path, *, target_hw: Tuple[int, int], target_depth: int):
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("torch is required for the RadFM worker.") from e

    img, vol = _load_nifti_volume(nifti_path)
    vol = _robust_normalize(vol)
    vol_chwhd = np.repeat(vol[np.newaxis, ...], 3, axis=0)
    tensor = torch.from_numpy(vol_chwhd).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(int(target_hw[0]), int(target_hw[1]), int(target_depth)),
        mode="trilinear",
        align_corners=False,
    )
    tensor = tensor[0]
    return img, vol, tensor


def _extract_choice_token(text: str, *, valid_choices: Sequence[str]) -> Optional[str]:
    raw = str(text or "")
    if not raw.strip():
        return None
    valid = tuple(str(choice).strip().upper() for choice in valid_choices if str(choice).strip())
    if not valid:
        return None
    choices_re = "|".join(re.escape(choice) for choice in valid)
    answer_re = re.compile(rf"\b(?:ANSWER|CHOICE|OPTION)\b\s*[:\-]?\s*({choices_re})\b", re.IGNORECASE)
    leading_choice_re = re.compile(rf"^\s*({choices_re})(?=[\s\.\)\:\-]|$)", re.IGNORECASE)
    m = answer_re.search(raw)
    if m:
        token = m.group(1).upper()
        return token if token in valid else None
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    for ln in reversed(lines[-10:]):
        token = ln.upper()
        if token in valid:
            return token
        m = leading_choice_re.match(ln)
        if m:
            token = m.group(1).upper()
            return token if token in valid else None
    return None


def _keyword_backoff_choice(text: str, *, valid_choices: Sequence[str]) -> Optional[str]:
    if not set(valid_choices).issubset({"A", "B", "C"}):
        return None
    raw = str(text or "").lower()
    if "no tumor" in raw or "no evidence of tumor" in raw:
        return "C" if "C" in valid_choices else None
    if "oligodendroglioma" in raw or "astrocytoma" in raw:
        return "B" if "B" in valid_choices else None
    if "glioblastoma" in raw:
        return "A" if "A" in valid_choices else None
    return None


def _sanitize_generated_text(raw_text: str, question: str) -> str:
    text = str(raw_text or "").replace("\r\n", "\n")
    text = _IMAGE_TOKEN_RE.sub("", text)
    prompt_markers = [
        question,
        "Write a brief rationale and end with a final line exactly formatted as: ANSWER: <A|B|C>.",
        "Reply with exactly one line in the format ANSWER: <A|B|C>.",
        "Reply with exactly one final line: ANSWER: X",
        "Choose exactly one option and reply with exactly one final line: ANSWER: X",
    ]
    for marker in prompt_markers:
        if marker and marker in text:
            text = text.rsplit(marker, 1)[-1]
    return text.strip()


def _has_structured_evidence(text: str) -> bool:
    raw = str(text or "")
    if "EVIDENCE:" not in raw.upper():
        return False
    upper = raw.upper()
    return all(key in upper for key in _EVIDENCE_KEYS)


def _voxel_to_ras(affine: np.ndarray, ijk: np.ndarray) -> np.ndarray:
    vec = np.asarray([float(ijk[0]), float(ijk[1]), float(ijk[2]), 1.0], dtype=float)
    ras = np.asarray(affine, dtype=float) @ vec
    return ras[:3]


def _evidence_from_original_mask(mask_ijk: np.ndarray, affine: np.ndarray) -> Optional[Dict[str, Any]]:
    coords = np.argwhere(mask_ijk > 0)
    if coords.size == 0:
        return None
    centroid = coords.mean(axis=0)
    nearest_idx = int(np.argmin(((coords - centroid) ** 2).sum(axis=1)))
    chosen = coords[nearest_idx].astype(float)
    ras = _voxel_to_ras(affine, chosen)
    return {
        "key_slice_mm": {
            "axial": float(ras[2]),
            "sagittal": float(ras[0]),
            "coronal": float(ras[1]),
        },
        "point_ras": [float(ras[0]), float(ras[1]), float(ras[2])],
    }


def _heuristic_mask_from_volume(vol_ijk: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi  # type: ignore

    vol = _robust_normalize(vol_ijk)
    smoothed = ndi.gaussian_filter(vol, sigma=(1.5, 1.5, 1.0))
    shape = np.asarray(smoothed.shape, dtype=int)
    margins = np.maximum((shape * np.asarray([0.08, 0.08, 0.06])).astype(int), np.asarray([2, 2, 1]))
    inner = np.zeros_like(smoothed, dtype=bool)
    i0, j0, k0 = [int(v) for v in margins]
    i1, j1, k1 = [int(shape[0] - margins[0]), int(shape[1] - margins[1]), int(shape[2] - margins[2])]
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        inner[...] = True
    else:
        inner[i0:i1, j0:j1, k0:k1] = True

    comp_mask = None
    thresholds = (0.997, 0.995, 0.992, 0.99, 0.985)
    center = (shape - 1) / 2.0
    for q in thresholds:
        thr = float(np.quantile(smoothed[inner], q))
        binary = (smoothed >= thr) & inner
        binary = ndi.binary_opening(binary, iterations=1)
        binary = ndi.binary_closing(binary, iterations=2)
        labeled, n = ndi.label(binary)
        if n <= 0:
            continue
        best_score = None
        best_label = None
        for label in range(1, int(n) + 1):
            coords = np.argwhere(labeled == label)
            if coords.shape[0] < 32:
                continue
            dist = float(np.linalg.norm((coords.mean(axis=0) - center) / np.maximum(shape, 1)))
            score = float(coords.shape[0]) / (1.0 + 5.0 * dist)
            if best_score is None or score > best_score:
                best_score = score
                best_label = label
        if best_label is not None:
            comp_mask = labeled == best_label
            break

    if comp_mask is None:
        comp_mask = np.zeros_like(smoothed, dtype=bool)
        center_idx = tuple(int(v) for v in center)
        comp_mask[center_idx] = True

    return comp_mask.astype(np.uint8)


def _format_evidence_block(evidence: Dict[str, Any]) -> str:
    key_slice_mm = evidence.get("key_slice_mm") or {}
    point_ras = evidence.get("point_ras") or [0.0, 0.0, 0.0]
    return "\n".join(
        [
            "EVIDENCE:",
            f"KEY_SLICE_AXIAL_MM: {float(key_slice_mm.get('axial', 0.0)):.4f}",
            f"KEY_SLICE_SAGITTAL_MM: {float(key_slice_mm.get('sagittal', 0.0)):.4f}",
            f"KEY_SLICE_CORONAL_MM: {float(key_slice_mm.get('coronal', 0.0)):.4f}",
            f"RSA: [{float(point_ras[0]):.4f}, {float(point_ras[1]):.4f}, {float(point_ras[2]):.4f}]",
        ]
    )


def _unwrap_state_dict(ckpt: Any) -> Any:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
    return ckpt


def _generate_from_model(model, text_tokenizer, prompt_text: str, vision_x, device: str, *, max_length: int, max_new_tokens: int) -> str:
    import torch

    with torch.no_grad():
        lang_x = text_tokenizer(prompt_text, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"].to(device)
        model.embedding_layer.flag = "Text"
        input_embedding, _ = model.embedding_layer(lang_x, vision_x)
        generation = model.lang_model.generate(inputs_embeds=input_embedding, max_new_tokens=max_new_tokens, top_k=50)
        generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True)
    return str(generated_texts[0] if generated_texts else "")


def _score_choice_logits_from_model(
    model,
    text_tokenizer,
    prompt_text: str,
    vision_x,
    device: str,
    *,
    max_length: int,
    valid_choices: Sequence[str],
) -> Tuple[Optional[str], Dict[str, float]]:
    import torch

    with torch.no_grad():
        lang_x = text_tokenizer(prompt_text, max_length=max_length, truncation=True, return_tensors="pt")["input_ids"].to(device)
        model.embedding_layer.flag = "Text"
        input_embedding, _ = model.embedding_layer(lang_x, vision_x)
        outputs = model.lang_model(inputs_embeds=input_embedding)
        logits = outputs.logits[:, -1, :]

    scores: Dict[str, float] = {}
    for choice in valid_choices:
        token_ids = text_tokenizer.encode(" " + choice, add_special_tokens=False)
        if len(token_ids) != 1:
            token_ids = text_tokenizer.encode(choice, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        scores[choice] = float(logits[0, token_ids[0]].item())

    if not scores:
        return None, {}
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    return best, scores


def run_radfm(request: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import torch
    except Exception as e:
        raise RuntimeError("The RadFM worker requires torch.") from e

    repo_root = Path(str(request["repo_root"])).expanduser().resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"RadFM repo root does not exist: {repo_root}")

    checkpoint_path = Path(str(request["checkpoint_path"])).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"RadFM checkpoint does not exist: {checkpoint_path}")

    nifti_path = Path(str(request["nifti_path"])).expanduser().resolve()
    if not nifti_path.exists():
        raise FileNotFoundError(f"Input NIfTI volume does not exist: {nifti_path}")

    question = str(request.get("question") or "")
    device = str(request.get("device") or "cuda")
    valid_choices = _normalize_valid_choices(request.get("valid_choices"))
    question_id = request.get("question_id")
    include_evidence = bool(request.get("include_evidence", False)) or "EVIDENCE:" in question.upper()
    segmentation_question = str(request.get("segmentation_question") or "")
    if not device.startswith("cuda"):
        raise RuntimeError("RadFM is intended for GPU inference. Use a CUDA device for this worker.")
    if not torch.cuda.is_available():
        raise RuntimeError("RadFM worker requested CUDA but torch.cuda.is_available() is False.")

    _patch_llama_from_pretrained(torch, device)
    language_dir = _resolve_language_dir(repo_root, request.get("language_dir"))
    MultiLLaMAForCausalLM = _import_radfm_model(repo_root)

    target_height = int(request.get("target_height") or 512)
    target_width = int(request.get("target_width") or 512)
    target_depth = _nearest_supported_depth(int(request.get("target_depth") or 32))
    max_model_length = int(request.get("model_max_length") or 2048)
    max_new_tokens = int(request.get("max_new_tokens") or 200)

    text_tokenizer, image_padding_tokens = _get_tokenizer(
        language_dir,
        max_img_size=int(request.get("max_img_size") or 10),
        image_num=int(request.get("image_num") or 32),
    )

    prompt_text = "<image>" + image_padding_tokens[0] + "</image>" + question
    img, vol_ijk, vision_x_single = _normalize_volume_for_radfm(
        nifti_path,
        target_hw=(target_height, target_width),
        target_depth=target_depth,
    )
    vision_x = vision_x_single.unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float16)

    model = MultiLLaMAForCausalLM(lang_model_path=str(language_dir))
    model = model.half().to(device)
    ckpt = torch.load(str(checkpoint_path), map_location=device)
    ckpt = _unwrap_state_dict(ckpt)
    load_result = model.load_state_dict(ckpt, strict=False)
    del ckpt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()

    raw_text = _generate_from_model(
        model,
        text_tokenizer,
        prompt_text,
        vision_x,
        device,
        max_length=max_model_length,
        max_new_tokens=max_new_tokens,
    )
    cleaned_text = _sanitize_generated_text(raw_text, question)
    choice = _extract_choice_token(cleaned_text, valid_choices=valid_choices)
    secondary_raw_text = None
    secondary_cleaned_text = None
    choice_logit_scores = None

    if valid_choices and choice is None:
        answer_spec = "|".join(valid_choices)
        valid_spec = ", ".join(valid_choices)
        short_question = (
            question.rstrip()
            + "\n\n"
            + f"Choose exactly one option from {{{valid_spec}}}. "
            + f"Reply with exactly one final line: ANSWER: <{answer_spec}>"
        )
        short_prompt_text = "<image>" + image_padding_tokens[0] + "</image>" + short_question
        secondary_raw_text = _generate_from_model(
            model,
            text_tokenizer,
            short_prompt_text,
            vision_x,
            device,
            max_length=max_model_length,
            max_new_tokens=32,
        )
        secondary_cleaned_text = _sanitize_generated_text(secondary_raw_text, short_question)
        choice = _extract_choice_token(secondary_cleaned_text, valid_choices=valid_choices)
        if choice is None:
            logit_prompt = "<image>" + image_padding_tokens[0] + "</image>" + question.rstrip() + "\n\nAnswer:"
            choice, choice_logit_scores = _score_choice_logits_from_model(
                model,
                text_tokenizer,
                logit_prompt,
                vision_x,
                device,
                max_length=max_model_length,
                valid_choices=valid_choices,
            )
            if choice is not None and (not secondary_cleaned_text.strip() or secondary_cleaned_text.strip() == '.'):
                secondary_cleaned_text = f"LOGIT_FALLBACK_SCORES: {choice_logit_scores}"
        if choice is None:
            choice = _keyword_backoff_choice(secondary_cleaned_text, valid_choices=valid_choices) or _keyword_backoff_choice(
                raw_text,
                valid_choices=valid_choices,
            )
        if choice is not None and secondary_cleaned_text.strip():
            cleaned_text = secondary_cleaned_text.strip()

    evidence_source = None
    evidence_payload = None

    if include_evidence and not _has_structured_evidence(cleaned_text):
        mask = _heuristic_mask_from_volume(vol_ijk)
        evidence_payload = _evidence_from_original_mask(mask, np.asarray(img.affine, dtype=float))
        if evidence_payload is not None:
            evidence_source = "heuristic_intensity_fallback"
            cleaned_text = cleaned_text.strip()
            if cleaned_text:
                cleaned_text += "\n\n"
            cleaned_text += _format_evidence_block(evidence_payload)

    if choice is not None and "ANSWER:" not in cleaned_text.upper():
        cleaned_text = cleaned_text.strip()
        if cleaned_text:
            cleaned_text += "\n"
        cleaned_text += f"ANSWER: {choice}"

    if not cleaned_text.strip():
        cleaned_text = raw_text.strip()

    missing_keys = list(getattr(load_result, "missing_keys", []) or [])
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []) or [])
    return {
        "ok": True,
        "worker": "radfm_worker",
        "repo_root": str(repo_root),
        "checkpoint_path": str(checkpoint_path),
        "language_dir": str(language_dir),
        "nifti_path": str(nifti_path),
        "device": device,
        "question": question,
        "question_id": question_id,
        "valid_choices": list(valid_choices),
        "include_evidence": include_evidence,
        "segmentation_question": segmentation_question,
        "text": cleaned_text,
        "raw_text": raw_text,
        "secondary_raw_text": secondary_raw_text,
        "secondary_cleaned_text": secondary_cleaned_text,
        "choice_logit_scores": choice_logit_scores,
        "parsed_choice": choice,
        "evidence_source": evidence_source,
        "evidence": evidence_payload,
        "load_state_dict": {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "n_missing": len(missing_keys),
            "n_unexpected": len(unexpected_keys),
        },
        "preprocess": {
            "vision_shape": [int(v) for v in vision_x.shape],
            "target_height": int(target_height),
            "target_width": int(target_width),
            "target_depth": int(target_depth),
            "original_shape": [int(v) for v in vol_ijk.shape],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Track C RadFM worker")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    req_path = Path(args.input_json).expanduser().resolve()
    out_path = Path(args.output_json).expanduser().resolve()
    request = _load_request(req_path)
    try:
        payload = run_radfm(request)
    except Exception as e:
        payload = {
            "ok": False,
            "worker": "radfm_worker",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        _write_response(out_path, payload)
        raise
    _write_response(out_path, payload)


if __name__ == "__main__":
    main()
