#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash shared/setup/run_release_checks.sh [--scope SCOPE]

Runs the source-only checks that are safe without datasets, a model, or
3D Slicer. PYTHON_BIN defaults to .venv-vllm/bin/python when present, then
falls back to python3.10.

Scopes:
  all        Check radiology and pathology runners (default)
  radiology  Check Track A/B and Track C runners
  pathology  Check QuPath pathology runners
EOF
}

scope="all"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scope)
      scope="${2:?--scope requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
case "$scope" in
  all|radiology|pathology) ;;
  *)
    echo "--scope must be all, radiology, or pathology: $scope" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv-vllm/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv-vllm/bin/python"
  else
    PYTHON_BIN="$(command -v python3.10 || true)"
  fi
fi
if [[ -z "${PYTHON_BIN:-}" || ( ! -x "$PYTHON_BIN" && -z "$(command -v "$PYTHON_BIN" 2>/dev/null)" ) ]]; then
  echo "Python 3.10 was not found. Set PYTHON_BIN to the reproduction environment." >&2
  exit 1
fi

"$PYTHON_BIN" "$SCRIPT_DIR/check_python_version.py"
"$PYTHON_BIN" -m pip check

REPO_ROOT="$REPO_ROOT" "$PYTHON_BIN" - <<'PY'
import ast
import os
from pathlib import Path

root = Path(os.environ["REPO_ROOT"])
excluded = {".git", ".pytest_cache", "runs"}
files = [
    path
    for path in root.rglob("*.py")
    if not excluded.intersection(path.relative_to(root).parts)
    and not any(part.startswith(".venv") for part in path.relative_to(root).parts)
]
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"Python syntax: {len(files)} files OK")
PY

shell_count=0
while IFS= read -r -d '' script; do
  bash -n "$script"
  shell_count=$((shell_count + 1))
done < <(
  find "$REPO_ROOT" \
    -path "$REPO_ROOT/.git" -prune -o \
    -path "$REPO_ROOT/runs" -prune -o \
    -path "$REPO_ROOT/.venv-vllm" -prune -o \
    -type f -name '*.sh' -print0
)
echo "Shell syntax: $shell_count files OK"

radiology_runners=(
  radiology/slicer_track_a_b/benchmark_UCSF_PDGM.py
  radiology/slicer_track_a_b/benchmark_NSCLC.py
  radiology/slicer_track_a_b/benchmark_LUMIERE.py
  radiology/track_c/benchmark_UCSF_PDGM_track_c.py
  radiology/track_c/benchmark_NSCLC_track_c.py
  radiology/track_c/benchmark_LUMIERE_track_c.py
)
pathology_runners=(
  pathology/benchmarks/benchmark_bracs.py
  pathology/benchmarks/benchmark_camelyon17.py
)
runners=()
if [[ "$scope" == "all" || "$scope" == "radiology" ]]; then
  runners+=("${radiology_runners[@]}")
fi
if [[ "$scope" == "all" || "$scope" == "pathology" ]]; then
  runners+=("${pathology_runners[@]}")
fi
for runner in "${runners[@]}"; do
  "$PYTHON_BIN" "$REPO_ROOT/$runner" --help >/dev/null
done
echo "Runner help/import checks ($scope): ${#runners[@]} files OK"

(
  cd "$REPO_ROOT"
  PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON_BIN" -m pytest -q -p no:cacheprovider shared/tests
)
