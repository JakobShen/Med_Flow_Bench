#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash shared/setup/create_python_env.sh PROFILE [ENV_DIR]

Profiles:
  track_a_b   3D Slicer Track A/B benchmark host
  track_c     Runtime-independent Track C API methods
  pathology   QuPath pathology benchmark host
  mcp         Optional Slicer MCP server
  server      Optional HTTP tool server
  vista3d     VISTA3D/MONAI service on a Linux CUDA machine
  dev         All host-side extras plus pytest

The script installs the local package in editable mode. It does not install
3D Slicer, QuPath, Java, Xvfb, CUDA drivers, model weights, M3D, or RadFM.

Overrides:
  PYTHON_BOOTSTRAP              Python 3.10 used to create the environment
  VISTA3D_TORCH_INDEX_URL      PyTorch wheel index for the vista3d profile
  VISTA3D_TORCH_VERSION        PyTorch version for the vista3d profile
  VISTA3D_TORCHVISION_VERSION  Torchvision version for the vista3d profile
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

PROFILE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-$(command -v python3.10 || command -v python3 || true)}"

case "$PROFILE" in
  track_a_b) DEFAULT_ENV="$REPO_ROOT/.venv-track_a_b" ;;
  track_c) DEFAULT_ENV="$REPO_ROOT/.venv-track_c" ;;
  pathology) DEFAULT_ENV="$REPO_ROOT/.venv-pathology" ;;
  mcp) DEFAULT_ENV="$REPO_ROOT/.venv-mcp" ;;
  server) DEFAULT_ENV="$REPO_ROOT/.venv-server" ;;
  vista3d) DEFAULT_ENV="$REPO_ROOT/.venv-vista3d" ;;
  dev) DEFAULT_ENV="$REPO_ROOT/.venv-dev" ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

ENV_DIR="${2:-$DEFAULT_ENV}"

if [[ -z "$PYTHON_BOOTSTRAP" ]]; then
  echo "Python 3.10 was not found. Set PYTHON_BOOTSTRAP explicitly." >&2
  exit 1
fi
"$PYTHON_BOOTSTRAP" "$REPO_ROOT/shared/setup/check_python_version.py"

if [[ "$PROFILE" == "vista3d" && "$(uname -s)" != "Linux" ]]; then
  echo "The vista3d profile requires a supported Linux accelerator host." >&2
  exit 1
fi

"$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"
PYTHON_BIN="$ENV_DIR/bin/python"
"$PYTHON_BIN" -m pip install --upgrade \
  "pip==26.1.2" \
  "setuptools==81.0.0" \
  "wheel==0.47.0"

case "$PROFILE" in
  track_a_b)
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/shared/requirements/track_a_b.txt"
    ;;
  track_c)
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/shared/requirements/track_c.txt"
    ;;
  pathology)
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/shared/requirements/pathology.txt"
    ;;
  mcp)
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/shared/requirements/agent.txt"
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/shared/requirements/mcp.txt"
    ;;
  server)
    "$PYTHON_BIN" -m pip install -e "$REPO_ROOT[agent,server]"
    ;;
  vista3d)
    TORCH_INDEX="${VISTA3D_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
    TORCH_VERSION="${VISTA3D_TORCH_VERSION:-2.5.1}"
    TORCHVISION_VERSION="${VISTA3D_TORCHVISION_VERSION:-0.20.1}"
    "$PYTHON_BIN" -m pip install \
      --index-url "$TORCH_INDEX" \
      "torch==$TORCH_VERSION" \
      "torchvision==$TORCHVISION_VERSION"
    "$PYTHON_BIN" -m pip install -r "$REPO_ROOT/shared/requirements/vista3d-server.txt"
    ;;
  dev)
    "$PYTHON_BIN" -m pip install -e "$REPO_ROOT[agent,qupath,mcp,server,dev]"
    ;;
esac

if [[ "$PROFILE" != "server" && "$PROFILE" != "dev" ]]; then
  "$PYTHON_BIN" -m pip install -e "$REPO_ROOT" --no-deps
fi

cat <<EOF
Environment created:
  profile: $PROFILE
  path:    $ENV_DIR

Activate it with:
  source "$ENV_DIR/bin/activate"
EOF
