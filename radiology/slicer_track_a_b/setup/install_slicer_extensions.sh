#!/usr/bin/env bash
set -euo pipefail

SLICER_EXECUTABLE="${SLICER_EXECUTABLE:-}"
XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/slicer-runtime-${UID:-user}}"
REQUIRED_EXTENSIONS_CSV="${SLICER_EXTENSIONS_CSV:-WebServer,SegmentEditorExtraEffects,SlicerElastix,QuantitativeReporting}"
OPTIONAL_EXTENSIONS_CSV="${SLICER_OPTIONAL_EXTENSIONS_CSV:-MONAILabel}"

if [[ -z "$SLICER_EXECUTABLE" ]]; then
  echo "Set SLICER_EXECUTABLE to the 3D Slicer executable." >&2
  exit 2
fi

if [[ ! -x "$SLICER_EXECUTABLE" ]]; then
  echo "Slicer executable not found: $SLICER_EXECUTABLE" >&2
  exit 2
fi

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
export XDG_RUNTIME_DIR
export REQUIRED_EXTENSIONS_CSV
export OPTIONAL_EXTENSIONS_CSV

tmp_py="$(mktemp /tmp/slicer-install-extensions-XXXXXX.py)"
cleanup() {
  rm -f "$tmp_py"
}
trap cleanup EXIT

cat >"$tmp_py" <<'PY'
import os
import sys
import importlib
from pathlib import Path

import slicer

required = [name.strip() for name in os.environ["REQUIRED_EXTENSIONS_CSV"].split(",") if name.strip()]
optional = [name.strip() for name in os.environ["OPTIONAL_EXTENSIONS_CSV"].split(",") if name.strip()]
em = slicer.app.extensionsManagerModel()
em.interactive = False

failed_required = []
failed_optional = []
for name, is_required in [(name, True) for name in required] + [(name, False) for name in optional]:
    if name == "WebServer":
        try:
            importlib.import_module("WebServer")
            print("[extensions] WebServer: provided by the Slicer application")
            continue
        except Exception:
            pass
    already = em.isExtensionInstalled(name)
    requirement = "required" if is_required else "optional"
    print(f"[extensions] {name}: {requirement}, installed={already}")
    if already:
        continue
    print(f"[extensions] installing {name}")
    ok = em.installExtensionFromServer(name, False)
    print(f"[extensions] install result {name}: {ok}")
    if not ok:
        if is_required:
            failed_required.append(name)
        else:
            failed_optional.append(name)
    slicer.app.processEvents()

# Shared filesystems may strip executable bits while Extension Manager extracts
# packages. Every Slicer CLI executable has a same-name XML description.
for name in em.installedExtensions:
    extension_path = Path(str(em.extensionInstallPath(name)))
    if not extension_path.is_dir():
        continue
    for description in extension_path.rglob("cli-modules/*.xml"):
        executable = description.with_suffix("")
        if executable.is_file() and executable.stat().st_mode & 0o111 == 0:
            executable.chmod(executable.stat().st_mode | 0o111)
            print(f"[extensions] restored executable mode: {executable}")

if failed_optional:
    print(f"[extensions] optional extensions unavailable: {failed_optional}", file=sys.stderr)
if failed_required:
    print(f"[extensions] required extensions failed: {failed_required}", file=sys.stderr)
    sys.exit(1)

print("[extensions] all required extensions are installed")
sys.exit(0)
PY

if [[ -n "${SLICER_EXTRA_LIBRARY_PATH:-}" ]]; then
  env LD_LIBRARY_PATH="$SLICER_EXTRA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$SLICER_EXECUTABLE" --no-splash --no-main-window --python-script "$tmp_py"
else
  "$SLICER_EXECUTABLE" --no-splash --no-main-window --python-script "$tmp_py"
fi
