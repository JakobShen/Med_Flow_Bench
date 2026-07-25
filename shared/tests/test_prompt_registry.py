from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any, Iterator, Tuple

from medflow_prompts import (
    PromptRegistryError,
    load_prompt,
    prompt_metadata,
    prompt_metadata_bundle,
    render_prompt,
    resolve_prompt_root,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = REPO_ROOT / "shared" / "prompts"
PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _prompt_files() -> list[Path]:
    return sorted(path for path in PROMPT_ROOT.rglob("*.json") if path.name != "schema.json")


def _template_leaves(value: Any, prefix: str = "templates") -> Iterator[Tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
        return
    if isinstance(value, list):
        yield prefix, "".join(value)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _template_leaves(child, f"{prefix}.{key}")
        return
    raise AssertionError(f"Unsupported prompt template at {prefix}: {type(value).__name__}")


class PromptRegistryTests(unittest.TestCase):
    def test_prompt_root_is_repository_directory(self) -> None:
        self.assertEqual(resolve_prompt_root(), PROMPT_ROOT.resolve())

    def test_all_prompt_files_load_and_render(self) -> None:
        files = _prompt_files()
        self.assertTrue(files)
        prompt_ids: set[str] = set()

        for path in files:
            ref = str(path.relative_to(PROMPT_ROOT))
            payload = load_prompt(ref)
            self.assertNotIn(payload["prompt_id"], prompt_ids)
            prompt_ids.add(payload["prompt_id"])
            self.assertRegex(payload["version"], SEMVER_RE)

            metadata = prompt_metadata(ref)
            self.assertEqual(metadata["prompt_id"], payload["prompt_id"])
            self.assertEqual(metadata["prompt_version"], payload["version"])
            self.assertEqual(len(metadata["prompt_sha256"]), 64)

            for template_key, template in _template_leaves(payload["templates"]):
                variables = {key: f"<{key}>" for key in PLACEHOLDER_RE.findall(template)}
                rendered = render_prompt(ref, template_key, **variables)
                self.assertIsNone(PLACEHOLDER_RE.search(rendered))

    def test_entrypoints_exist_and_reference_each_prompt(self) -> None:
        for path in _prompt_files():
            ref = str(path.relative_to(PROMPT_ROOT))
            payload = load_prompt(ref)
            source_texts = []
            for entrypoint in payload["entrypoints"]:
                source_path = REPO_ROOT / entrypoint
                self.assertTrue(source_path.is_file(), f"Missing prompt entrypoint: {entrypoint}")
                source_texts.append(source_path.read_text(encoding="utf-8"))
            self.assertTrue(
                any(ref in source for source in source_texts),
                f"No declared entrypoint directly references {ref}",
            )

    def test_all_benchmark_runners_record_prompt_metadata(self) -> None:
        runner_refs = {
            "radiology/slicer_track_a_b/benchmark_UCSF_PDGM.py": "3d_slicer/track_a_b/ucsf_pdgm.json",
            "radiology/slicer_track_a_b/benchmark_NSCLC.py": "3d_slicer/track_a_b/nsclc.json",
            "radiology/slicer_track_a_b/benchmark_LUMIERE.py": "3d_slicer/track_a_b/lumiere.json",
            "radiology/track_c/benchmark_UCSF_PDGM_track_c.py": "track_c/ucsf_pdgm.json",
            "radiology/track_c/benchmark_NSCLC_track_c.py": "track_c/nsclc.json",
            "radiology/track_c/benchmark_LUMIERE_track_c.py": "track_c/lumiere.json",
            "pathology/benchmarks/benchmark_bracs.py": "pathology/bracs.json",
            "pathology/benchmarks/benchmark_camelyon17.py": "pathology/camelyon17.json",
        }
        for runner, prompt_ref in runner_refs.items():
            source = (REPO_ROOT / runner).read_text(encoding="utf-8")
            self.assertIn(prompt_ref, source)
            self.assertIn("prompt_metadata", source)

    def test_prompt_metadata_bundle_is_ordered_and_deduplicated(self) -> None:
        first = "3d_slicer/track_a_b/ucsf_pdgm.json"
        second = "3d_slicer/toolpacks.json"
        bundle = prompt_metadata_bundle(first, second, first)
        self.assertEqual(
            [item["prompt_ref"] for item in bundle["definitions"]],
            [first, second],
        )

    def test_schema_references_resolve_to_canonical_schema(self) -> None:
        schema_path = (PROMPT_ROOT / "schema.json").resolve()
        for path in _prompt_files():
            payload = json.loads(path.read_text(encoding="utf-8"))
            declared_schema = (path.parent / payload["$schema"]).resolve()
            self.assertEqual(declared_schema, schema_path)

    def test_prompt_metadata_is_deterministic(self) -> None:
        ref = "3d_slicer/track_a_b/ucsf_pdgm.json"
        self.assertEqual(prompt_metadata(ref), prompt_metadata(ref))

    def test_prompt_reference_cannot_escape_root(self) -> None:
        with self.assertRaises(PromptRegistryError):
            load_prompt("../pyproject.toml")

    def test_schema_is_valid_json(self) -> None:
        payload = json.loads((PROMPT_ROOT / "schema.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["title"], "MedFlowBench Prompt Definition")
        self.assertIn("entrypoints", payload["required"])


if __name__ == "__main__":
    unittest.main()
