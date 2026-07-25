from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SLICER_ROOT = REPO_ROOT / "radiology" / "slicer_track_a_b"
TRACK_C_ROOT = REPO_ROOT / "radiology" / "track_c"
TRACK_C_ENGINE_ROOT = REPO_ROOT / "engines" / "radiology_track_c"
PATHOLOGY_ROOT = REPO_ROOT / "pathology"
NUMERIC_TRACK_RE = re.compile(
    r"\btrack(?:\s+|[_-])?(?:1(?:[/_-]?2)?|2|3)\b",
    re.IGNORECASE,
)
NON_RELEASE_ROOTS = {".git", ".pytest_cache", ".venv", ".venv-vllm", "runs"}


def _release_paths():
    for root_entry in REPO_ROOT.iterdir():
        if root_entry.name in NON_RELEASE_ROOTS:
            continue
        yield root_entry
        if root_entry.is_dir():
            yield from root_entry.rglob("*")


class RadiologyLayoutTests(unittest.TestCase):
    def test_slicer_and_track_c_are_sibling_directories(self) -> None:
        self.assertEqual(SLICER_ROOT.parent, TRACK_C_ROOT.parent)
        self.assertTrue((SLICER_ROOT / "runtime" / "slicer_bridge").is_dir())
        self.assertTrue((TRACK_C_ROOT / "workers").is_dir())
        self.assertFalse((REPO_ROOT / "3d_slicer").exists())

    def test_track_c_has_an_independent_engine_package(self) -> None:
        self.assertTrue((TRACK_C_ENGINE_ROOT / "__init__.py").is_file())
        self.assertFalse((REPO_ROOT / "engines" / "slicer_agent_engine" / "track_c").exists())

    def test_pathology_has_benchmarks_without_track_subdirectories(self) -> None:
        self.assertTrue((PATHOLOGY_ROOT / "benchmarks" / "benchmark_bracs.py").is_file())
        self.assertTrue((PATHOLOGY_ROOT / "benchmarks" / "benchmark_camelyon17.py").is_file())
        self.assertFalse((PATHOLOGY_ROOT / "track_a").exists())

    def test_track_c_entrypoints_do_not_import_slicer_runtime(self) -> None:
        forbidden = (
            "slicer_client",
            "slicer_launcher",
            "app.bootstrap",
            "ensure_slicer_runtime",
            "SLICER_BASE_URL",
            "slicer_bridge",
        )
        sources = list(TRACK_C_ROOT.rglob("*.py")) + list(TRACK_C_ENGINE_ROOT.rglob("*.py"))
        self.assertTrue(sources)
        for path in sources:
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{path} contains Slicer runtime token {token!r}")

    def test_track_c_entrypoints_export_paper_table_data(self) -> None:
        entrypoints = sorted(TRACK_C_ROOT.glob("benchmark_*_track_c.py"))
        self.assertEqual(len(entrypoints), 3)
        for path in entrypoints:
            source = path.read_text(encoding="utf-8")
            self.assertIn("write_paper_table_data(", source, str(path))

    def test_repository_uses_only_track_a_b_c_names(self) -> None:
        text_suffixes = {".example", ".json", ".md", ".py", ".sh", ".toml", ".txt"}
        for path in _release_paths():
            relative = str(path.relative_to(REPO_ROOT))
            self.assertIsNone(NUMERIC_TRACK_RE.search(relative), relative)
            if path.is_file() and path.suffix.lower() in text_suffixes:
                source = path.read_text(encoding="utf-8", errors="ignore")
                self.assertIsNone(NUMERIC_TRACK_RE.search(source), relative)


if __name__ == "__main__":
    unittest.main()
