#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash radiology/model_serving/create_vllm_env.sh [ENV_DIR]

Creates a Linux vLLM environment for the open-weight Table 1 models.
Default ENV_DIR: .venv-vllm

Overrides:
  PYTHON_BOOTSTRAP   Python 3.10.x with venv/ensurepip support (python3.10)
  VLLM_ENV_DIR      Alternative default environment directory
  VLLM_USE_LOCK     Use the audited full dependency lock (1)
  VLLM_REQUIREMENTS_LOCK  Alternative full lock file
  PIP_CACHE_DIR     Isolated pip cache directory (ENV_DIR/.pip-cache)
  PIP_INSTALL_ATTEMPTS  Complete pip install attempts (3)
  PIP_NETWORK_RETRIES   Per-attempt pip network retries (10)
  PIP_NETWORK_TIMEOUT   Per-request timeout in seconds (120)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "vLLM model serving requires a supported Linux accelerator environment." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-$(command -v python3.10 || command -v python3 || true)}"
ENV_DIR="${1:-${VLLM_ENV_DIR:-$REPO_ROOT/.venv-vllm}}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ENV_DIR/.pip-cache}"
VLLM_USE_LOCK="${VLLM_USE_LOCK:-1}"
VLLM_REQUIREMENTS_LOCK="${VLLM_REQUIREMENTS_LOCK:-$REPO_ROOT/shared/requirements/locks/vllm-linux-py310-cu13-20260725.txt}"

if [[ -z "$PYTHON_BOOTSTRAP" ]] ||
   { ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BOOTSTRAP" ]]; }; then
  echo "Python bootstrap executable not found: $PYTHON_BOOTSTRAP" >&2
  exit 1
fi

if ! "$PYTHON_BOOTSTRAP" "$REPO_ROOT/shared/setup/check_python_version.py"; then
  exit 1
fi
if ! "$PYTHON_BOOTSTRAP" - <<'PY'
import ensurepip
import venv
PY
then
  cat >&2 <<EOF
The selected Python cannot create an isolated environment:
  $PYTHON_BOOTSTRAP

Install the OS venv package (for example python3-venv on Ubuntu), load a
Python module that includes ensurepip, or point PYTHON_BOOTSTRAP at a Conda
Python:
  PYTHON_BOOTSTRAP=/path/to/conda/env/bin/python \\
    bash radiology/model_serving/create_vllm_env.sh "$ENV_DIR"
EOF
  exit 1
fi

"$PYTHON_BOOTSTRAP" -m venv "$ENV_DIR"

pip_install() {
  local attempt
  local max_attempts="${PIP_INSTALL_ATTEMPTS:-3}"
  local retry_count="${PIP_NETWORK_RETRIES:-10}"
  local timeout="${PIP_NETWORK_TIMEOUT:-120}"

  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    if "$ENV_DIR/bin/python" -m pip install \
      --progress-bar off \
      --retries "$retry_count" \
      --timeout "$timeout" \
      "$@"; then
      return 0
    fi
    if [[ "$attempt" -lt "$max_attempts" ]]; then
      echo "pip install failed (attempt $attempt/$max_attempts); retrying in 10 seconds." >&2
      sleep 10
    fi
  done
  echo "pip install failed after $max_attempts attempts." >&2
  return 1
}

pip_install --upgrade \
  "pip==26.1.2" \
  "setuptools==81.0.0" \
  "wheel==0.47.0"
if [[ "$VLLM_USE_LOCK" == "1" ]]; then
  if [[ ! -f "$VLLM_REQUIREMENTS_LOCK" ]]; then
    echo "Audited vLLM lock does not exist: $VLLM_REQUIREMENTS_LOCK" >&2
    exit 1
  fi
  pip_install -r "$VLLM_REQUIREMENTS_LOCK"
  pip_install --no-build-isolation --no-deps -e "$REPO_ROOT"
elif [[ "$VLLM_USE_LOCK" == "0" ]]; then
  pip_install -r "$REPO_ROOT/shared/requirements/vllm.txt"
  pip_install --no-build-isolation -e "$REPO_ROOT[agent,dev]"
else
  echo "VLLM_USE_LOCK must be 0 or 1: $VLLM_USE_LOCK" >&2
  exit 2
fi
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pytest --version

cat <<EOF
vLLM environment created at:
  $ENV_DIR

Use it with:
  export PYTHON_BIN="$ENV_DIR/bin/python"
  export VLLM_BIN="$ENV_DIR/bin/vllm"
  bash radiology/model_serving/serve_vllm.sh medgemma-1.5-4b-it
EOF
