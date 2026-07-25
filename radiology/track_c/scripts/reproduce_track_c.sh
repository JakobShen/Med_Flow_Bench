#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash radiology/track_c/scripts/reproduce_track_c.sh [OPTIONS]

Runs LUMIERE, UCSF-PDGM, and NSCLC Track C without 3D Slicer, then writes
track_c_results.csv/json. Configuration is read from .env.track_c by default.

Options:
  --model KEY             One model key from table1_models.json
  --models KEYS           Comma-separated model keys
  --model-set NAME        Manifest set: all, medgemma, open, or hosted
  --method NAME           compressed_figures (paper baseline) or table3_api
  --limit N               Units per dataset (default: 5; 0 means full)
  --no-resume             Rerun datasets even when a matching marker exists
  --help                  Show this message

Environment:
  TRACK_C_ENV_FILE        Configuration file (default: REPO_ROOT/.env.track_c)
  TRACK_C_MODEL_KEYS      Default model keys (default: medgemma-1.5-4b-it)
  TRACK_C_METHOD          Default method (default: compressed_figures)
  TRACK_C_LIMIT           Default per-dataset limit (default: 5)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="${TRACK_C_ENV_FILE:-$REPO_ROOT/.env.track_c}"

for argument in "$@"; do
  if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
    usage
    exit 0
  fi
done

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing Track C configuration: $ENV_FILE" >&2
  echo "Start with: cp shared/config/track_c.env.example .env.track_c" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

model_set="${TRACK_C_MODEL_SET:-}"
model_keys_override="${TRACK_C_MODEL_KEYS:-medgemma-1.5-4b-it}"
method="${TRACK_C_METHOD:-compressed_figures}"
limit="${TRACK_C_LIMIT:-5}"
resume="${TRACK_C_RESUME:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      model_keys_override="${2:?--model requires a value}"
      model_set=""
      shift 2
      ;;
    --models)
      model_keys_override="${2:?--models requires a value}"
      model_set=""
      shift 2
      ;;
    --model-set)
      model_set="${2:?--model-set requires a value}"
      model_keys_override=""
      shift 2
      ;;
    --method)
      method="${2:?--method requires a value}"
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

case "$method" in
  compressed_figures|table3_api) ;;
  *)
    echo "One-command API reproduction supports compressed_figures or table3_api, not: $method" >&2
    echo "Run M3D/RadFM directly with their dedicated worker environments; see shared/docs/track_c.md." >&2
    exit 2
    ;;
esac
if [[ ! "$limit" =~ ^[0-9]+$ ]]; then
  echo "--limit/TRACK_C_LIMIT must be a non-negative integer: $limit" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_HELPER="$REPO_ROOT/radiology/model_serving/vllm_config.py"
MODEL_CONFIG="$REPO_ROOT/shared/config/table1_models.json"
RESULTS_ROOT="${TRACK_C_RESULTS_ROOT:-$REPO_ROOT/runs/track_c}"
METHOD_RESULTS_ROOT="$RESULTS_ROOT/$method"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_CLIENT_HOST="${VLLM_CLIENT_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_START_TIMEOUT_SEC="${VLLM_START_TIMEOUT_SEC:-900}"
LUMIERE_LIMIT="${TRACK_C_LUMIERE_LIMIT:-$limit}"
UCSF_LIMIT="${TRACK_C_UCSF_LIMIT:-$limit}"
NSCLC_LIMIT="${TRACK_C_NSCLC_LIMIT:-$limit}"

for dataset_limit in "$LUMIERE_LIMIT" "$UCSF_LIMIT" "$NSCLC_LIMIT"; do
  if [[ ! "$dataset_limit" =~ ^[0-9]+$ ]]; then
    echo "Dataset-specific Track C limits must be non-negative integers." >&2
    exit 2
  fi
done

if [[ -n "$model_keys_override" ]]; then
  normalized_keys="${model_keys_override//,/ }"
  read -r -a model_keys <<<"$normalized_keys"
else
  mapfile -t model_keys < <("$PYTHON_BIN" "$CONFIG_HELPER" list --set "${model_set:-all}")
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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi
"$PYTHON_BIN" "$REPO_ROOT/shared/setup/check_python_version.py"
require_path UCSF_DATA_ROOT
require_path UCSF_METADATA_CSV
require_path NSCLC_DATA_ROOT
require_path NSCLC_METADATA_CSV
require_path NSCLC_LABELS_CSV
require_path LUMIERE_DATA_ROOT
require_path LUMIERE_RATINGS_CSV
require_path LUMIERE_COMPLETENESS_CSV
"$PYTHON_BIN" "$CONFIG_HELPER" validate >/dev/null

model_field() {
  "$PYTHON_BIN" "$CONFIG_HELPER" get "$1" "$2"
}

model_override_name() {
  printf 'TRACK_C_MODEL_ID_%s' "$(printf '%s' "$1" | tr '[:lower:].-' '[:upper:]__')"
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

mkdir -p "$METHOD_RESULTS_ROOT/_launcher"
SEQUENCE_LOG="$METHOD_RESULTS_ROOT/_launcher/sequence.log"
touch "$SEQUENCE_LOG"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$SEQUENCE_LOG"
}

vllm_pid=""
vllm_process_group=0
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
trap cleanup_vllm EXIT INT TERM

start_vllm() {
  local model_key="$1"
  local served_model="$2"
  local model_log="$METHOD_RESULTS_ROOT/_launcher/$model_key/vllm.log"
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
  "$PYTHON_BIN" - "$dataset_root" "$dataset" "$expected_count" "$method" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset = sys.argv[2]
expected = int(sys.argv[3])
method = sys.argv[4]
papers = list(root.rglob("Paper_Table_Data.json"))
if not papers:
    raise SystemExit(f"Benchmark completed without Paper_Table_Data.json under {root}")
paper = max(papers, key=lambda path: path.stat().st_mtime)
summary_path = paper.parent / "summary.json"
if not summary_path.is_file():
    raise SystemExit(f"Missing summary.json next to {paper}")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("method") != method:
    raise SystemExit(f"Expected method={method}, found {summary.get('method')} in {summary_path}")
count_key = "n_questions" if dataset == "lumiere" else "n_cases"
actual = int(summary.get(count_key, -1))
failed = int(summary.get("n_failed", 0) or 0)
if actual < 0:
    raise SystemExit(f"Missing {count_key} in {summary_path}")
if failed:
    raise SystemExit(f"{dataset} completed with {failed} failed unit(s); inspect {summary_path.parent}")
if expected and actual != expected:
    raise SystemExit(f"{dataset} prepared {actual} eligible units, expected {expected}")
if dataset == "nsclc":
    segment_metrics = summary.get("segment_metrics") or {}
    n_segment_gt = int(segment_metrics.get("n_cases_with_segment_gt", 0) or 0)
    if not n_segment_gt:
        print(
            "WARNING: NSCLC has no usable public DICOM SEG or aligned NIfTI tumor masks. "
            "Task metrics are valid, but strict/localization values are zero "
            "placeholders with no evidence-scoring denominator.",
            file=sys.stderr,
        )
print(f"validated {dataset}: {actual} eligible units ({paper})")
PY
}

run_dataset() {
  local model_key="$1"
  local provider="$2"
  local model="$3"
  local dataset="$4"
  local dataset_root="$METHOD_RESULTS_ROOT/$model_key/$dataset"
  local marker="$dataset_root/.complete"
  local dataset_limit
  case "$dataset" in
    lumiere) dataset_limit="$LUMIERE_LIMIT" ;;
    ucsf_pdgm) dataset_limit="$UCSF_LIMIT" ;;
    nsclc) dataset_limit="$NSCLC_LIMIT" ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
  esac
  local expected_count="$dataset_limit"
  case "$dataset" in
    lumiere) [[ "$dataset_limit" == "0" ]] && expected_count=139 ;;
    ucsf_pdgm) [[ "$dataset_limit" == "0" ]] && expected_count=495 ;;
    nsclc) [[ "$dataset_limit" == "0" ]] && expected_count=162 ;;
  esac
  if [[ "$resume" == "1" && -f "$marker" ]] &&
     grep -Fx "method=$method" "$marker" >/dev/null 2>&1 &&
     grep -Fx "limit=$dataset_limit" "$marker" >/dev/null 2>&1; then
    validate_dataset_output "$dataset_root" "$dataset" "$expected_count"
    log "Skipping completed $method/$model_key/$dataset"
    return
  fi
  mkdir -p "$dataset_root"
  log "Running $method/$model_key/$dataset"

  case "$dataset" in
    lumiere)
      "$PYTHON_BIN" "$REPO_ROOT/radiology/track_c/benchmark_LUMIERE_track_c.py" \
        --provider "$provider" \
        --model "$model" \
        --method "$method" \
        --data_root "$LUMIERE_DATA_ROOT" \
        --ratings_csv "$LUMIERE_RATINGS_CSV" \
        --datacompleteness_csv "$LUMIERE_COMPLETENESS_CSV" \
        --balance_rano_labels \
        --min_scorable_evidence_fields 2 \
        --min_label_count_before_balance 10 \
        --limit_question "$LUMIERE_LIMIT" \
        --log_dir "$dataset_root"
      ;;
    ucsf_pdgm)
      "$PYTHON_BIN" "$REPO_ROOT/radiology/track_c/benchmark_UCSF_PDGM_track_c.py" \
        --provider "$provider" \
        --model "$model" \
        --method "$method" \
        --data_root "$UCSF_DATA_ROOT" \
        --csv_path "$UCSF_METADATA_CSV" \
        --balance_diagnosis_subtypes \
        --limit "$UCSF_LIMIT" \
        --log_dir "$dataset_root"
      ;;
    nsclc)
      "$PYTHON_BIN" "$REPO_ROOT/radiology/track_c/benchmark_NSCLC_track_c.py" \
        --provider "$provider" \
        --model "$model" \
        --method "$method" \
        --data_root "$NSCLC_DATA_ROOT" \
        --metadata_csv "$NSCLC_METADATA_CSV" \
        --labels_csv "$NSCLC_LABELS_CSV" \
        --limit "$NSCLC_LIMIT" \
        --log_dir "$dataset_root"
      ;;
  esac

  validate_dataset_output "$dataset_root" "$dataset" "$expected_count"
  {
    printf 'method=%s\n' "$method"
    printf 'limit=%s\n' "$dataset_limit"
    printf 'completed_at=%s\n' "$(date -Iseconds)"
  } >"$marker"
  log "Completed $method/$model_key/$dataset"
}

model_outputs_are_complete() {
  local model_key="$1"
  local dataset dataset_limit marker
  [[ "$resume" == "1" ]] || return 1
  for dataset in lumiere ucsf_pdgm nsclc; do
    case "$dataset" in
      lumiere) dataset_limit="$LUMIERE_LIMIT" ;;
      ucsf_pdgm) dataset_limit="$UCSF_LIMIT" ;;
      nsclc) dataset_limit="$NSCLC_LIMIT" ;;
    esac
    marker="$METHOD_RESULTS_ROOT/$model_key/$dataset/.complete"
    [[ -f "$marker" ]] || return 1
    grep -Fx "method=$method" "$marker" >/dev/null 2>&1 || return 1
    grep -Fx "limit=$dataset_limit" "$marker" >/dev/null 2>&1 || return 1
  done
}

log "Track C method: $method"
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

  if model_outputs_are_complete "$model_key"; then
    cleanup_vllm
  elif [[ "$runtime" == "vllm" ]]; then
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

"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_track_c.py" \
  --results-root "$METHOD_RESULTS_ROOT" \
  --method "$method" \
  --config "$MODEL_CONFIG" \
  --models "${model_keys[@]}"
log "Track C reproduction complete: $METHOD_RESULTS_ROOT/track_c_results.csv"
