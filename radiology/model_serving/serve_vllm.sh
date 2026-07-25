#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash radiology/model_serving/serve_vllm.sh MODEL_KEY

Starts one Table 1 open-weight model as an OpenAI-compatible vLLM server.
Run vllm_config.py list --set open for every open-weight MODEL_KEY or
vllm_config.py list --set medgemma for only the MedGemma keys.

Common overrides:
  VLLM_MODEL_PATH                  Local weights path instead of Hugging Face
  VLLM_MODEL_PATH_<MODEL_KEY>      Per-model path; key is uppercased with _ separators
  VLLM_HOST / VLLM_PORT            Listen address and port (127.0.0.1:8000)
  VLLM_API_KEY                     Server API key (EMPTY)
  VLLM_TENSOR_PARALLEL_SIZE        Number of GPUs (1)
  VLLM_GPU_MEMORY_UTILIZATION      Per-GPU fraction (0.90)
  VLLM_MAX_MODEL_LEN               Override manifest context length
  VLLM_EXTRA_ARGS                  Additional whitespace-separated vLLM flags
  MEDFLOW_VLLM_CACHE_DIR           Runtime compile/cache directory
  VLLM_NO_USAGE_STATS             Disable vLLM telemetry and its home write (1)

On Slurm, runtime caches default to node-local TMPDIR to avoid home-directory
quota failures. When nvcc or ninja is unavailable, the script defaults
VLLM_USE_FLASHINFER_SAMPLER=0 so the standard PyTorch sampler is used.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_HELPER="$SCRIPT_DIR/vllm_config.py"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_BIN="${VLLM_BIN:-vllm}"
vllm_executable="$VLLM_BIN"
MODEL_KEY="$1"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" "$REPO_ROOT/shared/setup/check_python_version.py"
if ! command -v "$VLLM_BIN" >/dev/null 2>&1 && [[ ! -x "$VLLM_BIN" ]]; then
  echo "vLLM executable not found: $VLLM_BIN" >&2
  echo "Create the environment with radiology/model_serving/create_vllm_env.sh." >&2
  exit 1
fi

runtime="$("$PYTHON_BIN" "$CONFIG_HELPER" get "$MODEL_KEY" runtime)"
if [[ "$runtime" != "vllm" ]]; then
  echo "Model $MODEL_KEY uses runtime=$runtime, not vLLM." >&2
  exit 2
fi

source_model="$("$PYTHON_BIN" "$CONFIG_HELPER" get "$MODEL_KEY" source_model)"
served_model="$("$PYTHON_BIN" "$CONFIG_HELPER" get "$MODEL_KEY" served_model)"
env_suffix="$(printf '%s' "$MODEL_KEY" | tr '[:lower:].-' '[:upper:]__')"
model_path_variable="VLLM_MODEL_PATH_${env_suffix}"
model_path="${VLLM_MODEL_PATH:-${!model_path_variable:-$source_model}}"

host="${VLLM_HOST:-127.0.0.1}"
port="${VLLM_PORT:-8000}"
api_key="${VLLM_API_KEY:-EMPTY}"
tensor_parallel_size="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
image_limit="${VLLM_IMAGE_LIMIT:-64}"

cache_dir="${MEDFLOW_VLLM_CACHE_DIR:-}"
if [[ -z "$cache_dir" && -n "${SLURM_JOB_ID:-}" ]]; then
  cache_dir="${TMPDIR:-/tmp}/${USER:-user}/medflow-vllm-${SLURM_JOB_ID}"
fi
if [[ -n "$cache_dir" ]]; then
  mkdir -p "$cache_dir"/{xdg,torchinductor,triton,vllm}
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_dir/xdg}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$cache_dir/torchinductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_dir/triton}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$cache_dir/vllm}"
  echo "vLLM runtime cache: $cache_dir"
fi

if [[ -z "${VLLM_USE_FLASHINFER_SAMPLER+x}" ]]; then
  if ! command -v nvcc >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    echo "nvcc or ninja not found; using the PyTorch sampler (VLLM_USE_FLASHINFER_SAMPLER=0)."
  fi
fi
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"

mapfile -t manifest_args < <("$PYTHON_BIN" "$CONFIG_HELPER" serve-args "$MODEL_KEY")
if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  filtered_args=()
  skip_next=0
  for arg in "${manifest_args[@]}"; do
    if (( skip_next )); then
      skip_next=0
      continue
    fi
    if [[ "$arg" == "--max-model-len" ]]; then
      skip_next=1
      continue
    fi
    filtered_args+=("$arg")
  done
  manifest_args=("${filtered_args[@]}" --max-model-len "$VLLM_MAX_MODEL_LEN")
fi

extra_args=()
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<<"$VLLM_EXTRA_ARGS"
fi

echo "Starting $MODEL_KEY ($served_model) from $model_path"
echo "OpenAI-compatible endpoint: http://$host:$port/v1"

# vLLM 0.21 warns about every unknown exported VLLM_* variable. The launcher
# variables have already been consumed above, so do not leak them into vLLM's
# own environment. Keep genuine vLLM runtime variables such as
# VLLM_NO_USAGE_STATS, VLLM_CACHE_ROOT, and VLLM_USE_FLASHINFER_SAMPLER.
unset \
  VLLM_API_KEY \
  VLLM_BIN \
  VLLM_CLIENT_HOST \
  VLLM_ENV_DIR \
  VLLM_EXTRA_ARGS \
  VLLM_GPU_MEMORY_UTILIZATION \
  VLLM_HOST \
  VLLM_IMAGE_LIMIT \
  VLLM_MAX_MODEL_LEN \
  VLLM_MODEL_PATH \
  VLLM_PORT \
  VLLM_REQUIREMENTS_LOCK \
  VLLM_START_TIMEOUT_SEC \
  VLLM_TENSOR_PARALLEL_SIZE \
  VLLM_USE_LOCK
while IFS= read -r variable_name; do
  unset "$variable_name"
done < <(compgen -v VLLM_MODEL_PATH_)

exec "$vllm_executable" serve "$model_path" \
  --served-model-name "$served_model" \
  --host "$host" \
  --port "$port" \
  --api-key "$api_key" \
  --tensor-parallel-size "$tensor_parallel_size" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --limit-mm-per-prompt "{\"image\":$image_limit}" \
  --trust-remote-code \
  "${manifest_args[@]}" \
  "${extra_args[@]}"
