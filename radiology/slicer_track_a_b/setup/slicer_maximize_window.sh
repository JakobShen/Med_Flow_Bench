#!/usr/bin/env bash
set -euo pipefail

# Resize/maximize the currently running Slicer GUI from a launcher .sh.
# Required: SLICER_BASE_URL must point to a Slicer WebServer with exec enabled.
# Optional:
#   PYTHON_BIN=/path/to/python
#   SLICER_MAIN_WINDOW_SIZE=2400x1800
#   SLICER_MAIN_WINDOW_MAXIMIZED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SLICER_BASE_URL="${SLICER_BASE_URL:-http://127.0.0.1:2016}"
SLICER_MAIN_WINDOW_SIZE="${SLICER_MAIN_WINDOW_SIZE:-2400x1800}"
SLICER_MAIN_WINDOW_MAXIMIZED="${SLICER_MAIN_WINDOW_MAXIMIZED:-1}"

export PYTHONPATH="$REPO_ROOT/engines:${PYTHONPATH:-}"
export SLICER_BASE_URL
export SLICER_MAIN_WINDOW_SIZE
export SLICER_MAIN_WINDOW_MAXIMIZED

"$PYTHON_BIN" - <<'PY'
import json
import os
from slicer_agent_engine.slicer_client import SlicerClient


def _parse_size(raw: str) -> tuple[int, int] | None:
    value = str(raw or "").strip().lower().replace(",", "x")
    if "x" not in value:
        return None
    left, right = value.split("x", 1)
    try:
        width = int(left.strip())
        height = int(right.strip())
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


target = _parse_size(os.environ.get("SLICER_MAIN_WINDOW_SIZE", ""))
target_payload = list(target) if target is not None else None
maximize = str(os.environ.get("SLICER_MAIN_WINDOW_MAXIMIZED", "1")).strip().lower() in {"1", "true", "yes", "on"}
maximize_py = "True" if maximize else "False"

source = f"""
import slicer

def _size(widget):
    if widget is None:
        return None
    size_attr = getattr(widget, "size", None)
    size = size_attr() if callable(size_attr) else size_attr
    if size is None:
        return None
    return [int(size.width()), int(size.height())]

main_window = slicer.util.mainWindow()
before = _size(main_window)
target = {json.dumps(target_payload)}
if main_window is not None:
    main_window.show()
    if target:
        main_window.resize(int(target[0]), int(target[1]))
    if {maximize_py}:
        main_window.showMaximized()
    for _ in range(8):
        slicer.app.processEvents()

lm = slicer.app.layoutManager()
view_sizes = {{}}
if lm is not None:
    for key, name in [("axial", "Red"), ("sagittal", "Yellow"), ("coronal", "Green")]:
        sw = lm.sliceWidget(name)
        sv = sw.sliceView() if sw else None
        view_sizes[key] = _size(sv)

__execResult = {{
    "ok": main_window is not None,
    "target": target,
    "maximize": {maximize_py},
    "before": before,
    "after": _size(main_window),
    "view_sizes": view_sizes,
}}
"""

client = SlicerClient(base_url=os.environ["SLICER_BASE_URL"], timeout_s=60.0)
print(json.dumps(client.exec_python(source, timeout_s=60.0), indent=2, sort_keys=True), flush=True)
PY
