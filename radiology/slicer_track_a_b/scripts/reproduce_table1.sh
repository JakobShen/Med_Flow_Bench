#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh [OPTIONS]

Runs the three viewer-native radiology benchmarks used in Table 1 and writes
table1_results.csv/json. Configuration is read from .env.table1 by default.

Options:
  --model-set NAME       Manifest set: all, medgemma, open, or hosted
  --models KEYS          Comma-separated manifest model keys
  --limit N              Cases/questions per dataset (default: 5; 0 means full)
  --no-resume            Rerun datasets even when a .complete marker exists
  --help                 Show this message

Environment:
  TABLE1_ENV_FILE        Configuration file (default: REPO_ROOT/.env.table1)
  TABLE1_MODEL_SET       Default model set (default: all)
  TABLE1_MODEL_KEYS      Alternative comma/space-separated model keys
  TABLE1_LIMIT           Default per-dataset limit (default: 5)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${TABLE1_ENV_FILE:-$REPO_ROOT/.env.table1}"

for argument in "$@"; do
  if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
    usage
    exit 0
  fi
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing Table 1 configuration: $ENV_FILE" >&2
  echo "Start with: cp shared/config/table1.env.example .env.table1" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

model_set="${TABLE1_MODEL_SET:-all}"
model_keys_override="${TABLE1_MODEL_KEYS:-}"
limit="${TABLE1_LIMIT:-5}"
resume="${TABLE1_RESUME:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-set)
      model_set="${2:?--model-set requires a value}"
      shift 2
      ;;
    --models)
      model_keys_override="${2:?--models requires a value}"
      shift 2
      ;;
    --limit)
      limit="${2:?--limit requires a value}"
      shift 2
      ;;
    --no-resume)
      resume=0
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$limit" =~ ^[0-9]+$ ]]; then
  echo "--limit/TABLE1_LIMIT must be a non-negative integer: $limit" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_HELPER="$REPO_ROOT/radiology/model_serving/vllm_config.py"
MODEL_CONFIG="$REPO_ROOT/shared/config/table1_models.json"
RESULTS_ROOT="${TABLE1_RESULTS_ROOT:-$REPO_ROOT/runs/table1}"
SLICER_BASE_URL="${SLICER_BASE_URL:-http://127.0.0.1:2016}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_CLIENT_HOST="${VLLM_CLIENT_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_START_TIMEOUT_SEC="${VLLM_START_TIMEOUT_SEC:-900}"
LUMIERE_LIMIT="${TABLE1_LUMIERE_LIMIT:-$limit}"
UCSF_LIMIT="${TABLE1_UCSF_LIMIT:-$limit}"
NSCLC_LIMIT="${TABLE1_NSCLC_LIMIT:-$limit}"

for dataset_limit in "$LUMIERE_LIMIT" "$UCSF_LIMIT" "$NSCLC_LIMIT"; do
  if [[ ! "$dataset_limit" =~ ^[0-9]+$ ]]; then
    echo "Dataset-specific Table 1 limits must be non-negative integers." >&2
    exit 2
  fi
done

if [[ -n "$model_keys_override" ]]; then
  normalized_keys="${model_keys_override//,/ }"
  read -r -a model_keys <<<"$normalized_keys"
else
  mapfile -t model_keys < <("$PYTHON_BIN" "$CONFIG_HELPER" list --set "$model_set")
fi
if [[ ${#model_keys[@]} -eq 0 ]]; then
  echo "No models selected." >&2
  exit 2
fi

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required configuration: $name" >&2
    exit 1
  fi
}

require_path() {
  local name="$1"
  require_value "$name"
  if [[ ! -e "${!name}" ]]; then
    echo "Configured path does not exist: $name=${!name}" >&2
    exit 1
  fi
}

require_path SLICER_EXECUTABLE
require_path UCSF_DATA_ROOT
require_path UCSF_METADATA_CSV
require_path NSCLC_DATA_ROOT
require_path NSCLC_METADATA_CSV
require_path NSCLC_LABELS_CSV
require_path LUMIERE_DATA_ROOT
require_path LUMIERE_RATINGS_CSV
require_path LUMIERE_COMPLETENESS_CSV
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" "$REPO_ROOT/shared/setup/check_python_version.py"
"$PYTHON_BIN" "$CONFIG_HELPER" validate >/dev/null

model_field() {
  "$PYTHON_BIN" "$CONFIG_HELPER" get "$1" "$2"
}

model_override_name() {
  printf 'TABLE1_MODEL_ID_%s' "$(printf '%s' "$1" | tr '[:lower:].-' '[:upper:]__')"
}

check_hosted_credentials() {
  local provider="$1"
  case "$provider" in
    openai)
      require_value OPENAI_API_KEY
      ;;
    gemini)
      if [[ -z "${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}" && -z "${GOOGLE_CLOUD_PROJECT:-}" ]]; then
        echo "Gemini requires GEMINI_API_KEY/GOOGLE_API_KEY or Vertex AI project credentials." >&2
        exit 1
      fi
      ;;
    claude)
      if [[ -z "${ANTHROPIC_API_KEY:-}" && -z "${ANTHROPIC_VERTEX_PROJECT_ID:-${GOOGLE_CLOUD_PROJECT:-}}" ]]; then
        echo "Claude requires ANTHROPIC_API_KEY or Vertex AI project credentials." >&2
        exit 1
      fi
      ;;
  esac
}

for model_key in "${model_keys[@]}"; do
  runtime="$(model_field "$model_key" runtime)"
  provider="$(model_field "$model_key" provider)"
  if [[ "$runtime" == "hosted" ]]; then
    check_hosted_credentials "$provider"
  elif [[ "$runtime" == "vllm" ]]; then
    require_value VLLM_BIN
    if ! command -v "$VLLM_BIN" >/dev/null 2>&1 && [[ ! -x "$VLLM_BIN" ]]; then
      echo "vLLM executable not found: $VLLM_BIN" >&2
      exit 1
    fi
  fi
done

mkdir -p "$RESULTS_ROOT/_launcher"
SEQUENCE_LOG="$RESULTS_ROOT/_launcher/sequence.log"
touch "$SEQUENCE_LOG"
export SLICER_LAUNCH_LOG_DIR="${SLICER_LAUNCH_LOG_DIR:-$RESULTS_ROOT/_launcher}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SEQUENCE_LOG"
}

vllm_pid=""
vllm_process_group=0
xvfb_pid=""
cleanup_vllm() {
  if [[ -n "$vllm_pid" ]]; then
    if [[ "$vllm_process_group" == "1" ]]; then
      kill -TERM -- "-$vllm_pid" 2>/dev/null || true
    elif kill -0 "$vllm_pid" 2>/dev/null; then
      kill -TERM "$vllm_pid" 2>/dev/null || true
    fi
    wait "$vllm_pid" 2>/dev/null || true
  fi
  vllm_pid=""
  vllm_process_group=0
}
cleanup() {
  cleanup_vllm
  if [[ -n "$xvfb_pid" ]] && kill -0 "$xvfb_pid" 2>/dev/null; then
    kill "$xvfb_pid" 2>/dev/null || true
    wait "$xvfb_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -z "${DISPLAY:-}" ]]; then
  XVFB_BIN="${XVFB_BIN:-$(command -v Xvfb || true)}"
  if [[ -z "$XVFB_BIN" ]]; then
    echo "DISPLAY is unset and Xvfb was not found. Install Xvfb or export DISPLAY." >&2
    exit 1
  fi
  export DISPLAY="${TABLE1_DISPLAY:-:99}"
  if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "$XDG_RUNTIME_DIR" || ! -w "$XDG_RUNTIME_DIR" ]]; then
    export XDG_RUNTIME_DIR="$RESULTS_ROOT/_launcher/xdg-runtime"
  fi
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
  xvfb_args=("$DISPLAY" -screen 0 1920x1080x24 +extension GLX +render -nolisten tcp)
  if [[ -n "${XVFB_EXTRA_ARGS:-}" ]]; then
    read -r -a xvfb_extra_args <<<"$XVFB_EXTRA_ARGS"
    xvfb_args+=("${xvfb_extra_args[@]}")
  fi
  if [[ -n "${XVFB_LIB_DIR:-}" ]]; then
    env LD_LIBRARY_PATH="$XVFB_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$XVFB_BIN" "${xvfb_args[@]}" >"$RESULTS_ROOT/_launcher/xvfb.log" 2>&1 &
  else
    "$XVFB_BIN" "${xvfb_args[@]}" >"$RESULTS_ROOT/_launcher/xvfb.log" 2>&1 &
  fi
  xvfb_pid=$!
  xvfb_ready=0
  for _ in $(seq 1 30); do
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
      break
    fi
    if ! command -v xdpyinfo >/dev/null 2>&1 ||
       xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
      xvfb_ready=1
      break
    fi
    sleep 1
  done
  if [[ "$xvfb_ready" != "1" ]]; then
    echo "Xvfb failed to start; inspect $RESULTS_ROOT/_launcher/xvfb.log" >&2
    tail -n 80 "$RESULTS_ROOT/_launcher/xvfb.log" >&2 || true
    exit 1
  fi
  log "Started Xvfb on $DISPLAY"
fi

run_benchmark_host() {
  if [[ -n "${SLICER_EXTRA_LIBRARY_PATH:-}" ]]; then
    LD_LIBRARY_PATH="$SLICER_EXTRA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$@"
  else
    "$@"
  fi
}

start_vllm() {
  local model_key="$1"
  local served_model="$2"
  local model_log="$RESULTS_ROOT/_launcher/$model_key/vllm.log"
  mkdir -p "$(dirname "$model_log")"
  cleanup_vllm
  log "Starting vLLM for $model_key"
  if command -v setsid >/dev/null 2>&1; then
    setsid env \
      PYTHON_BIN="$PYTHON_BIN" \
      VLLM_BIN="$VLLM_BIN" \
      VLLM_HOST="$VLLM_HOST" \
      VLLM_PORT="$VLLM_PORT" \
      VLLM_API_KEY="$VLLM_API_KEY" \
      bash "$REPO_ROOT/radiology/model_serving/serve_vllm.sh" "$model_key" \
      >"$model_log" 2>&1 &
    vllm_process_group=1
  else
    env \
      PYTHON_BIN="$PYTHON_BIN" \
      VLLM_BIN="$VLLM_BIN" \
      VLLM_HOST="$VLLM_HOST" \
      VLLM_PORT="$VLLM_PORT" \
      VLLM_API_KEY="$VLLM_API_KEY" \
      bash "$REPO_ROOT/radiology/model_serving/serve_vllm.sh" "$model_key" \
      >"$model_log" 2>&1 &
    vllm_process_group=0
  fi
  vllm_pid=$!

  local base_url="http://$VLLM_CLIENT_HOST:$VLLM_PORT"
  local deadline=$((SECONDS + VLLM_START_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$vllm_pid" 2>/dev/null; then
      echo "vLLM exited while starting $model_key. Last log lines:" >&2
      tail -n 80 "$model_log" >&2 || true
      exit 1
    fi
    if curl -fsS "$base_url/health" >/dev/null 2>&1 &&
       curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" "$base_url/v1/models" 2>/dev/null |
         grep -F "$served_model" >/dev/null; then
      export LOCAL_VLLM_BASE_URL="$base_url/v1"
      export LOCAL_VLLM_API_KEY="$VLLM_API_KEY"
      export LOCAL_VLLM_MODEL="$served_model"
      "$PYTHON_BIN" - "$LOCAL_VLLM_BASE_URL" "$LOCAL_VLLM_API_KEY" "$served_model" <<'PY'
import sys
from openai import OpenAI

base_url, api_key, model = sys.argv[1:]
response = OpenAI(base_url=base_url, api_key=api_key).responses.create(
    model=model,
    input="Reply with exactly OK.",
    max_output_tokens=256,
)
if not getattr(response, "output", None):
    raise RuntimeError("vLLM Responses API smoke request returned no output")
print("vLLM Responses API ready:", model)
PY
      log "vLLM ready for $model_key"
      return
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM model $model_key. Last log lines:" >&2
  tail -n 80 "$model_log" >&2 || true
  exit 1
}

validate_dataset_output() {
  local dataset_root="$1"
  local dataset="$2"
  local expected_count="$3"
  "$PYTHON_BIN" - "$dataset_root" "$dataset" "$expected_count" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset = sys.argv[2]
expected = int(sys.argv[3])
papers = list(root.rglob("Paper_Table_Data.json"))
if not papers:
    raise SystemExit(f"Benchmark completed without Paper_Table_Data.json under {root}")
paper = max(papers, key=lambda path: path.stat().st_mtime)
summary_path = paper.parent / "summary.json"
if not summary_path.is_file():
    raise SystemExit(f"Missing summary.json next to {paper}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
count_key = "n_questions" if dataset == "lumiere" else "n_cases"
actual = int(summary.get(count_key, -1))
if actual < 0:
    raise SystemExit(f"Missing {count_key} in {summary_path}")
failed = int(summary.get("n_failed", 0) or 0)
if failed:
    raise SystemExit(
        f"{dataset} completed with {failed} failed unit(s); refusing to write "
        f"a completion marker. Inspect {summary_path.parent}."
    )
if expected and actual != expected:
    raise SystemExit(
        f"{dataset} prepared {actual} eligible units, expected {expected}. "
        "Check the dataset release and metadata paths."
    )
print(f"validated {dataset}: {actual} eligible units ({paper})")
PY
}

run_dataset() {
  local model_key="$1"
  local provider="$2"
  local model="$3"
  local dataset="$4"
  local dataset_root="$RESULTS_ROOT/$model_key/$dataset"
  local marker="$dataset_root/.complete"
  local dataset_limit
  case "$dataset" in
    lumiere) dataset_limit="$LUMIERE_LIMIT" ;;
    ucsf_pdgm) dataset_limit="$UCSF_LIMIT" ;;
    nsclc) dataset_limit="$NSCLC_LIMIT" ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac
  if [[ "$resume" == "1" && -f "$marker" ]] &&
     grep -Fx "limit=$dataset_limit" "$marker" >/dev/null 2>&1; then
    log "Skipping completed $model_key/$dataset"
    return
  fi
  mkdir -p "$dataset_root"
  log "Running $model_key/$dataset"

  case "$dataset" in
    lumiere)
      run_benchmark_host "$PYTHON_BIN" "$REPO_ROOT/radiology/slicer_track_a_b/benchmark_LUMIERE.py" \
        --provider "$provider" \
        --model "$model" \
        --data_root "$LUMIERE_DATA_ROOT" \
        --ratings_csv "$LUMIERE_RATINGS_CSV" \
        --datacompleteness_csv "$LUMIERE_COMPLETENESS_CSV" \
        --lumiere_image_space registered \
        --balance_rano_labels \
        --min_scorable_evidence_fields 2 \
        --min_label_count_before_balance 10 \
        --max_rounds 20 \
        --limit_question "$LUMIERE_LIMIT" \
        --base_url "$SLICER_BASE_URL" \
        --slicer_executable "$SLICER_EXECUTABLE" \
        --log_dir "$dataset_root"
      ;;
    ucsf_pdgm)
      run_benchmark_host "$PYTHON_BIN" "$REPO_ROOT/radiology/slicer_track_a_b/benchmark_UCSF_PDGM.py" \
        --provider "$provider" \
        --model "$model" \
        --data_root "$UCSF_DATA_ROOT" \
        --csv_path "$UCSF_METADATA_CSV" \
        --balance_diagnosis_subtypes \
        --max_rounds 20 \
        --limit "$UCSF_LIMIT" \
        --base_url "$SLICER_BASE_URL" \
        --slicer_executable "$SLICER_EXECUTABLE" \
        --log_dir "$dataset_root"
      ;;
    nsclc)
      run_benchmark_host "$PYTHON_BIN" "$REPO_ROOT/radiology/slicer_track_a_b/benchmark_NSCLC.py" \
        --provider "$provider" \
        --model "$model" \
        --data_root "$NSCLC_DATA_ROOT" \
        --metadata_csv "$NSCLC_METADATA_CSV" \
        --labels_csv "$NSCLC_LABELS_CSV" \
        --max_rounds 20 \
        --limit "$NSCLC_LIMIT" \
        --base_url "$SLICER_BASE_URL" \
        --slicer_executable "$SLICER_EXECUTABLE" \
        --log_dir "$dataset_root"
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 2
      ;;
  esac

  local expected_count="$dataset_limit"
  case "$dataset" in
    lumiere)
      [[ "$dataset_limit" == "0" ]] && expected_count=139
      ;;
    ucsf_pdgm)
      [[ "$dataset_limit" == "0" ]] && expected_count=495
      ;;
    nsclc)
      [[ "$dataset_limit" == "0" ]] && expected_count=162
      ;;
  esac
  validate_dataset_output "$dataset_root" "$dataset" "$expected_count"
  {
    printf 'limit=%s\n' "$dataset_limit"
    printf 'completed_at=%s\n' "$(date -Iseconds)"
  } >"$marker"
  log "Completed $model_key/$dataset"
}

log "Selected models: ${model_keys[*]}"
log "Dataset limits: LUMIERE=$LUMIERE_LIMIT UCSF-PDGM=$UCSF_LIMIT NSCLC=$NSCLC_LIMIT"
for model_key in "${model_keys[@]}"; do
  runtime="$(model_field "$model_key" runtime)"
  provider="$(model_field "$model_key" provider)"
  model="$(model_field "$model_key" model)"
  override_variable="$(model_override_name "$model_key")"
  if [[ -n "${!override_variable:-}" ]]; then
    model="${!override_variable}"
  fi

  if [[ "$runtime" == "vllm" ]]; then
    served_model="$(model_field "$model_key" served_model)"
    start_vllm "$model_key" "$served_model"
    model="$served_model"
  else
    cleanup_vllm
  fi

  run_dataset "$model_key" "$provider" "$model" lumiere
  run_dataset "$model_key" "$provider" "$model" ucsf_pdgm
  run_dataset "$model_key" "$provider" "$model" nsclc
done
cleanup_vllm

"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_table1.py" \
  --results-root "$RESULTS_ROOT" \
  --config "$MODEL_CONFIG" \
  --models "${model_keys[@]}"
log "Table 1 reproduction complete: $RESULTS_ROOT/table1_results.csv"
