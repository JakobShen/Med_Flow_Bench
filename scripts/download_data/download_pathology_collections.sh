#!/usr/bin/env bash
set -euo pipefail

# Download public pathology benchmark collections.
#
# Modes:
#   smoke   Check that the public endpoints are reachable.
#   resume  Download or resume CAMELYON17 and BRACS.
#
# Example:
#   DATA_ROOT=/path/to/data bash scripts/download_data/download_pathology_collections.sh resume

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "resume" ]]; then
  echo "Usage: $0 {smoke|resume}" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-$PWD/data}"
WORK_ROOT="${WORK_ROOT:-$PWD/.download_work/pathology}"
TMP_ROOT="$WORK_ROOT/tmp"
TOOL_ROOT="$WORK_ROOT/tools"
AWS_CLI_ZIP_URL="https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip"

CAMELYON17_S3_URL="s3://camelyon-dataset/CAMELYON17/"
CAMELYON17_OUT="$DATA_ROOT/CAMELYON17"

BRACS_FTP_ROOT="ftp://histoimage.na.icar.cnr.it"
BRACS_OUT="$DATA_ROOT/BRACS"
BRACS_MANIFEST="$TMP_ROOT/bracs.paths.txt"

AWS_ZIP="$TMP_ROOT/awscliv2.zip"
AWS_EXTRACT="$TMP_ROOT/awscli"
AWS_INSTALL_DIR="$TOOL_ROOT/aws-cli"
AWS_BIN_DIR="$TOOL_ROOT/bin"
AWS_BIN="$AWS_BIN_DIR/aws"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
  log "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"
}

preflight() {
  mkdir -p "$DATA_ROOT" "$TMP_ROOT" "$TOOL_ROOT" "$CAMELYON17_OUT" "$BRACS_OUT"
  require_cmd curl
  require_cmd wget
  require_cmd unzip
  touch "$DATA_ROOT/.write_test"
  rm -f "$DATA_ROOT/.write_test"
  log "DATA_ROOT=$DATA_ROOT"
  log "WORK_ROOT=$WORK_ROOT"
}

prepare_aws_cli() {
  if command -v aws >/dev/null 2>&1; then
    AWS_BIN="$(command -v aws)"
    log "Using system AWS CLI: $AWS_BIN"
    "$AWS_BIN" --version
    return 0
  fi

  if [[ -x "$AWS_BIN" ]]; then
    log "Using cached AWS CLI: $AWS_BIN"
    "$AWS_BIN" --version
    return 0
  fi

  log "Installing AWS CLI under $TOOL_ROOT"
  rm -rf "$AWS_EXTRACT"
  mkdir -p "$AWS_EXTRACT" "$AWS_INSTALL_DIR" "$AWS_BIN_DIR"
  curl -fL "$AWS_CLI_ZIP_URL" -o "$AWS_ZIP"
  unzip -qo "$AWS_ZIP" -d "$AWS_EXTRACT"
  "$AWS_EXTRACT/aws/install" --update -i "$AWS_INSTALL_DIR" -b "$AWS_BIN_DIR"
  [[ -x "$AWS_BIN" ]] || fail "AWS CLI was not installed at $AWS_BIN"
  "$AWS_BIN" --version
}

build_bracs_manifest() {
  cat >"$BRACS_MANIFEST" <<'EOF'
BRACS.xlsx
BRACS_RoI/
BRACS_WSI/
BRACS_WSI_Annotations/
EOF
}

probe_bracs() {
  build_bracs_manifest
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ "$path" == */ ]]; then
      wget --spider -q "$BRACS_FTP_ROOT/$path" || fail "BRACS path is not reachable: $path"
    else
      wget --spider -q "$BRACS_FTP_ROOT/$path" || fail "BRACS file is not reachable: $path"
    fi
    log "Reachable: $BRACS_FTP_ROOT/$path"
  done <"$BRACS_MANIFEST"
}

download_bracs() {
  build_bracs_manifest
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    if [[ "$path" == */ ]]; then
      mkdir -p "$BRACS_OUT/$path"
      wget -c -r -np -nH --cut-dirs=0 --reject "index.html*" -P "$BRACS_OUT" "$BRACS_FTP_ROOT/$path"
    else
      wget -c -O "$BRACS_OUT/$path" "$BRACS_FTP_ROOT/$path"
    fi
  done <"$BRACS_MANIFEST"
}

probe_camelyon17() {
  prepare_aws_cli
  "$AWS_BIN" s3 ls --no-sign-request "$CAMELYON17_S3_URL" | head -n 20
}

download_camelyon17() {
  prepare_aws_cli
  "$AWS_BIN" s3 sync --no-sign-request "$CAMELYON17_S3_URL" "$CAMELYON17_OUT"
}

preflight

case "$MODE" in
  smoke)
    probe_camelyon17
    probe_bracs
    log "Smoke checks completed."
    ;;
  resume)
    download_camelyon17
    download_bracs
    ;;
esac

log "Done."
