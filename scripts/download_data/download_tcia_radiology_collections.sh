#!/usr/bin/env bash
set -euo pipefail

# Download public radiology benchmark collections.
#
# Modes:
#   smoke    Check that the public endpoints are reachable.
#   resume   Download or resume NSCLC-Radiogenomics, UCSF-PDGM, and LUMIERE.
#   organize Run lightweight post-download organization steps.
#
# Example:
#   DATA_ROOT=/path/to/data bash scripts/download_data/download_tcia_radiology_collections.sh resume

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "resume" && "$MODE" != "organize" ]]; then
  echo "Usage: $0 {smoke|resume|organize}" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-$PWD/data}"
WORK_ROOT="${WORK_ROOT:-$PWD/.download_work/tcia_radiology}"
APP_ROOT="$WORK_ROOT/apps"
TMP_ROOT="$WORK_ROOT/tmp"
LOG_ROOT="$WORK_ROOT/log"

NBIA_DEB_URL="https://github.com/CBIIT/NBIA-TCIA/releases/download/DR-4_4_3-TCIA-20240916-1/nbia-data-retriever_4.4.3-1_amd64.deb"
NSCLC_MANIFEST_URL="https://www.cancerimagingarchive.net/wp-content/uploads/NSCLC_Radiogenomics-6-1-21-Version-4.tcia"

UCSF_RELEASE_ASSETS_URL="https://github.com/data-nih/tcia/releases/expanded_assets/ucsf-pdgm"
UCSF_METADATA_CSV="${UCSF_METADATA_CSV:-}"

FIGSHARE_API="https://api.figshare.com/v2"
LUMIERE_COLLECTION_ID="5904905"

NBIA_ROOT="$APP_ROOT/nbia"
NBIA_DEB="$TMP_ROOT/nbia-data-retriever_4.4.3-1_amd64.deb"
NBIA_BIN=""

NSCLC_MANIFEST="$TMP_ROOT/nsclc-radiogenomics.tcia"
NSCLC_OUT="$DATA_ROOT/nsclc-radiogenomics"
UCSF_OUT="$DATA_ROOT/ucsf-pdgm"
UCSF_STAGE="$UCSF_OUT/.flat"
UCSF_URLS="$TMP_ROOT/ucsf-pdgm.urls.txt"
LUMIERE_OUT="$DATA_ROOT/LUMIERE"
LUMIERE_MANIFEST="$TMP_ROOT/lumiere.files.tsv"

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
  mkdir -p "$DATA_ROOT" "$APP_ROOT" "$TMP_ROOT" "$LOG_ROOT" "$NSCLC_OUT" "$UCSF_STAGE" "$LUMIERE_OUT"
  require_cmd python3
  touch "$DATA_ROOT/.write_test"
  rm -f "$DATA_ROOT/.write_test"
  log "DATA_ROOT=$DATA_ROOT"
  log "WORK_ROOT=$WORK_ROOT"
}

download_url() {
  local url="$1"
  local target="$2"
  mkdir -p "$(dirname "$target")"
  if [[ -s "$target" ]]; then
    log "Already exists: $target"
    return 0
  fi
  log "Downloading $url"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 --summary-interval=60 --file-allocation=none -d "$(dirname "$target")" -o "$(basename "$target")" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$target" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 10 --retry-delay 10 -C - -o "$target" "$url"
  else
    fail "Need aria2c, wget, or curl for downloads."
  fi
}

http_probe() {
  local url="$1"
  python3 - "$url" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
request = urllib.request.Request(url, headers={"User-Agent": "dataset-download-helper"})
with urllib.request.urlopen(request, timeout=60) as response:
    print(f"{response.status} {url}")
PY
}

prepare_nbia() {
  if [[ -n "$NBIA_BIN" && -x "$NBIA_BIN" ]]; then
    return 0
  fi
  download_url "$NBIA_DEB_URL" "$NBIA_DEB"
  rm -rf "$NBIA_ROOT"
  mkdir -p "$NBIA_ROOT"
  if command -v dpkg-deb >/dev/null 2>&1; then
    dpkg-deb -x "$NBIA_DEB" "$NBIA_ROOT"
  elif command -v ar >/dev/null 2>&1; then
    (
      cd "$NBIA_ROOT"
      ar x "$NBIA_DEB"
      tar -xf data.tar.*
      rm -f control.tar.* data.tar.* debian-binary
    )
  else
    fail "Need dpkg-deb or ar to unpack NBIA Data Retriever."
  fi
  NBIA_BIN="$(find "$NBIA_ROOT" -type f \( -name 'nbia-data-retriever' -o -name 'NBIADataRetriever' \) | head -n 1)"
  [[ -n "$NBIA_BIN" ]] || fail "Could not find NBIA executable after unpacking."
  chmod +x "$NBIA_BIN"
  log "NBIA_BIN=$NBIA_BIN"
}

download_nsclc() {
  require_cmd java
  prepare_nbia
  download_url "$NSCLC_MANIFEST_URL" "$NSCLC_MANIFEST"
  log "Downloading NSCLC-Radiogenomics into $NSCLC_OUT"
  "$NBIA_BIN" --cli "$NSCLC_MANIFEST" -d "$NSCLC_OUT" --agree-to-license <<<"M"
}

build_ucsf_urls() {
  log "Fetching UCSF-PDGM release asset list."
  python3 - "$UCSF_RELEASE_ASSETS_URL" "$UCSF_URLS" <<'PY'
import html
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

page_url = sys.argv[1]
out_path = Path(sys.argv[2])
request = urllib.request.Request(page_url, headers={"User-Agent": "dataset-download-helper"})
with urllib.request.urlopen(request, timeout=120) as response:
    page = response.read().decode("utf-8", "ignore")

urls = set()
for href in re.findall(r'href="([^"]+/releases/download/ucsf-pdgm/[^"]+)"', page):
    href = html.unescape(href)
    if href.startswith("/"):
        href = urllib.parse.urljoin("https://github.com", href)
    urls.add(href)

if not urls:
    raise SystemExit("No UCSF-PDGM release assets found.")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(sorted(urls)) + "\n", encoding="utf-8")
print(f"Wrote {len(urls)} UCSF-PDGM URLs to {out_path}")
PY
}

download_ucsf() {
  build_ucsf_urls
  while IFS= read -r url; do
    [[ -n "$url" ]] || continue
    download_url "$url" "$UCSF_STAGE/$(basename "$url")"
  done <"$UCSF_URLS"
  organize_ucsf
}

organize_ucsf() {
  mkdir -p "$UCSF_OUT"
  if [[ -n "$UCSF_METADATA_CSV" ]]; then
    if [[ -f "$UCSF_METADATA_CSV" ]]; then
      ln -sf "$UCSF_METADATA_CSV" "$UCSF_OUT/$(basename "$UCSF_METADATA_CSV")"
      log "Linked UCSF metadata CSV into $UCSF_OUT"
    else
      fail "UCSF_METADATA_CSV was set but does not exist: $UCSF_METADATA_CSV"
    fi
  fi
  log "UCSF-PDGM assets are in $UCSF_STAGE"
}

build_lumiere_manifest() {
  log "Fetching LUMIERE Figshare file manifest."
  python3 - "$FIGSHARE_API" "$LUMIERE_COLLECTION_ID" "$LUMIERE_MANIFEST" <<'PY'
import json
import sys
import urllib.request
from pathlib import Path

api_root, collection_id, manifest_path = sys.argv[1:4]
headers = {"User-Agent": "dataset-download-helper"}

def get_json(url):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))

articles = get_json(f"{api_root}/collections/{collection_id}/articles")
rows = []
for article in articles:
    detail = get_json(f"{api_root}/articles/{article['id']}")
    for item in detail.get("files", []):
        name = item.get("name")
        url = item.get("download_url")
        if name and url:
            rows.append((name, url))

if not rows:
    raise SystemExit("No LUMIERE Figshare files found.")

path = Path(manifest_path)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text("".join(f"{name}\t{url}\n" for name, url in rows), encoding="utf-8")
print(f"Wrote {len(rows)} LUMIERE file URLs to {path}")
PY
}

download_lumiere() {
  build_lumiere_manifest
  while IFS=$'\t' read -r name url; do
    [[ -n "$name" && -n "$url" ]] || continue
    download_url "$url" "$LUMIERE_OUT/$name"
  done <"$LUMIERE_MANIFEST"
}

run_smoke() {
  http_probe "$NBIA_DEB_URL"
  http_probe "$NSCLC_MANIFEST_URL"
  build_ucsf_urls
  head -n 3 "$UCSF_URLS"
  build_lumiere_manifest
  head -n 3 "$LUMIERE_MANIFEST"
  log "Smoke checks completed."
}

preflight

case "$MODE" in
  smoke)
    run_smoke
    ;;
  resume)
    download_nsclc
    download_ucsf
    download_lumiere
    ;;
  organize)
    organize_ucsf
    ;;
esac

log "Done."
