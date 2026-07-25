from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


class PromptRegistryError(ValueError):
    pass


_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
_PROMPT_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SCHEMA_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+$")
_SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def resolve_prompt_root() -> Path:
    configured = str(os.environ.get("MEDFLOW_PROMPT_DIR") or "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise PromptRegistryError(f"MEDFLOW_PROMPT_DIR is not a directory: {root}")
        return root

    candidates = []
    for parent in Path(__file__).resolve().parents:
        candidates.extend((parent / "shared" / "prompts", parent / "prompts"))
    candidates.extend((Path.cwd() / "shared" / "prompts", Path.cwd() / "prompts"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise PromptRegistryError(
        "Could not locate the repository shared/prompts directory. "
        "Run from the repository checkout or set MEDFLOW_PROMPT_DIR."
    )


def _resolve_prompt_path(prompt_ref: str) -> Path:
    ref = Path(str(prompt_ref).strip())
    if ref.is_absolute() or ".." in ref.parts:
        raise PromptRegistryError(f"Prompt reference must be a safe relative path: {prompt_ref!r}")
    if ref.suffix != ".json":
        ref = ref.with_suffix(".json")
    root = resolve_prompt_root()
    path = (root / ref).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PromptRegistryError(f"Prompt reference escapes the prompt root: {prompt_ref!r}") from exc
    if not path.is_file():
        raise PromptRegistryError(f"Prompt file does not exist: {path}")
    return path


@lru_cache(maxsize=64)
def load_prompt(prompt_ref: str) -> dict[str, Any]:
    path = _resolve_prompt_path(prompt_ref)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PromptRegistryError(f"Invalid prompt JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PromptRegistryError(f"Prompt file must contain a JSON object: {path}")
    for key in (
        "$schema",
        "schema_version",
        "prompt_id",
        "version",
        "benchmark",
        "track",
        "description",
        "entrypoints",
        "templates",
    ):
        if key not in payload:
            raise PromptRegistryError(f"Prompt file {path} is missing required key {key!r}")
    if not _SCHEMA_VERSION_RE.fullmatch(str(payload["schema_version"])):
        raise PromptRegistryError(f"Prompt file {path} has invalid schema_version")
    if not _PROMPT_ID_RE.fullmatch(str(payload["prompt_id"])):
        raise PromptRegistryError(f"Prompt file {path} has invalid prompt_id")
    if not _SEMVER_RE.fullmatch(str(payload["version"])):
        raise PromptRegistryError(f"Prompt file {path} has invalid semantic version")
    entrypoints = payload["entrypoints"]
    if (
        not isinstance(entrypoints, list)
        or not entrypoints
        or not all(isinstance(item, str) and item.strip() for item in entrypoints)
        or len(set(entrypoints)) != len(entrypoints)
    ):
        raise PromptRegistryError(f"Prompt file {path} must define unique non-empty entrypoints")
    if not isinstance(payload["templates"], dict):
        raise PromptRegistryError(f"Prompt file {path} must define templates as an object")
    _validate_template_node(payload["templates"], path="templates")
    return payload


def _validate_template_node(value: Any, *, path: str) -> None:
    if isinstance(value, str):
        return
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise PromptRegistryError(f"Prompt template {path!r} must contain only strings")
        return
    if isinstance(value, Mapping) and value:
        for key, child in value.items():
            _validate_template_node(child, path=f"{path}.{key}")
        return
    raise PromptRegistryError(
        f"Prompt template {path!r} must be a string, list of strings, or non-empty object"
    )


def _lookup_template(payload: Mapping[str, Any], template_key: str) -> Any:
    value: Any = payload
    for part in str(template_key).split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise PromptRegistryError(
                f"Prompt {payload.get('prompt_id')!r} has no template field {template_key!r}"
            )
        value = value[part]
    return value


def _coerce_template_text(value: Any, *, template_key: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    raise PromptRegistryError(f"Prompt template {template_key!r} must be a string or list of strings")


def render_prompt(prompt_ref: str, template_key: str, /, **variables: Any) -> str:
    payload = load_prompt(prompt_ref)
    template = _coerce_template_text(
        _lookup_template(payload, template_key),
        template_key=template_key,
    )
    required = set(_PLACEHOLDER_RE.findall(template))
    missing = sorted(name for name in required if name not in variables)
    if missing:
        raise PromptRegistryError(
            f"Prompt {payload['prompt_id']!r} template {template_key!r} "
            f"is missing variables: {', '.join(missing)}"
        )
    rendered = _PLACEHOLDER_RE.sub(lambda match: str(variables[match.group(1)]), template)
    unresolved = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if unresolved:
        raise PromptRegistryError(
            f"Prompt {payload['prompt_id']!r} left unresolved variables: {', '.join(unresolved)}"
        )
    return rendered


def prompt_metadata(prompt_ref: str) -> dict[str, str]:
    payload = load_prompt(prompt_ref)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "prompt_id": str(payload["prompt_id"]),
        "prompt_version": str(payload["version"]),
        "prompt_schema_version": str(payload["schema_version"]),
        "prompt_ref": str(Path(prompt_ref).with_suffix(".json")),
        "prompt_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def prompt_metadata_bundle(*prompt_refs: str) -> dict[str, list[dict[str, str]]]:
    if not prompt_refs:
        raise PromptRegistryError("At least one prompt reference is required")
    definitions = []
    seen = set()
    for prompt_ref in prompt_refs:
        metadata = prompt_metadata(prompt_ref)
        ref = metadata["prompt_ref"]
        if ref in seen:
            continue
        seen.add(ref)
        definitions.append(metadata)
    return {"definitions": definitions}
