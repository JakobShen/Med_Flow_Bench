#!/usr/bin/env bash
set -euo pipefail

# Download the public LUMIERE Figshare release and extract the native,
# unregistered image-space files used by the benchmark data preparation flow.
#
# Examples:
#   bash scripts/download_data/download_lumiere_unregistered.sh
#   OUT_DIR=/path/to/LUMIERE_unregistered EXTRACT_MODE=full bash scripts/download_data/download_lumiere_unregistered.sh
#   EXTRACT_MODE=none bash scripts/download_data/download_lumiere_unregistered.sh

OUT_DIR="${OUT_DIR:-$PWD/data/LUMIERE_unregistered}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$OUT_DIR/.downloads}"
EXTRACT_MODE="${EXTRACT_MODE:-native_only}"  # native_only, full, or none
KEEP_ZIP="${KEEP_ZIP:-1}"

IMAGING_URL="https://ndownloader.figshare.com/files/38249697"
IMAGING_FILE="Imaging-v202211.zip"
IMAGING_MD5="b2708d297badd245c8c8cbbb91fdda40"

README_URL="https://ndownloader.figshare.com/files/37983597"
README_FILE="LUMIERE-readme.pdf"
README_MD5="1d8cd1e273727ca6c4abc1d69bd8a0de"

RATING_URL="https://ndownloader.figshare.com/files/38369741"
RATING_FILE="LUMIERE-ExpertRating-v202211.csv"
RATING_MD5="3f196e118aa5390952ff8397aa3d25ed"

COMPLETENESS_URL="https://ndownloader.figshare.com/files/37575523"
COMPLETENESS_FILE="LUMIERE-datacompleteness.csv"
COMPLETENESS_MD5="3be2baf9b7fe060cca03b7965ae5513f"

MRINFO_URL="https://ndownloader.figshare.com/files/37691442"
MRINFO_FILE="LUMIERE-MRinfo.csv"
MRINFO_MD5="ddea9919d1b229e0d3735fd8c5dac6c4"

mkdir -p "$OUT_DIR" "$DOWNLOAD_DIR"
LOG_FILE="$OUT_DIR/download_lumiere_unregistered_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "OUT_DIR=$OUT_DIR"
echo "DOWNLOAD_DIR=$DOWNLOAD_DIR"
echo "EXTRACT_MODE=$EXTRACT_MODE"
echo "LOG_FILE=$LOG_FILE"

calc_md5() {
  local path="$1"
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$path" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$path"
  else
    echo "No md5sum/md5 command found." >&2
    return 1
  fi
}

verify_md5() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(calc_md5 "$path")"
  if [[ "$actual" != "$expected" ]]; then
    echo "MD5 mismatch for $path" >&2
    echo "  expected: $expected" >&2
    echo "  actual:   $actual" >&2
    return 1
  fi
}

download_file() {
  local url="$1"
  local name="$2"
  local expected_md5="$3"
  local target="$DOWNLOAD_DIR/$name"

  if [[ -s "$target" ]]; then
    if verify_md5 "$target" "$expected_md5"; then
      echo "Already downloaded and verified: $target"
      return 0
    fi
    echo "Existing file failed MD5 check; re-downloading: $target"
    rm -f "$target"
  fi

  echo "Downloading $name"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 --summary-interval=60 --file-allocation=none -d "$DOWNLOAD_DIR" -o "$name" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -c -O "$target" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 10 --retry-delay 10 -C - -o "$target" "$url"
  else
    echo "Need aria2c, wget, or curl to download files." >&2
    return 1
  fi

  verify_md5 "$target" "$expected_md5"
  echo "Verified: $target"
}

download_file "$README_URL" "$README_FILE" "$README_MD5"
download_file "$RATING_URL" "$RATING_FILE" "$RATING_MD5"
download_file "$COMPLETENESS_URL" "$COMPLETENESS_FILE" "$COMPLETENESS_MD5"
download_file "$MRINFO_URL" "$MRINFO_FILE" "$MRINFO_MD5"
download_file "$IMAGING_URL" "$IMAGING_FILE" "$IMAGING_MD5"

cp -f "$DOWNLOAD_DIR/$README_FILE" "$OUT_DIR/$README_FILE"
cp -f "$DOWNLOAD_DIR/$RATING_FILE" "$OUT_DIR/$RATING_FILE"
cp -f "$DOWNLOAD_DIR/$COMPLETENESS_FILE" "$OUT_DIR/$COMPLETENESS_FILE"
cp -f "$DOWNLOAD_DIR/$MRINFO_FILE" "$OUT_DIR/$MRINFO_FILE"

case "$EXTRACT_MODE" in
  none)
    echo "Skipping extraction because EXTRACT_MODE=none."
    ;;
  full)
    echo "Extracting full LUMIERE imaging zip into $OUT_DIR"
    unzip -n "$DOWNLOAD_DIR/$IMAGING_FILE" -d "$OUT_DIR"
    ;;
  native_only)
    echo "Extracting native/unregistered LUMIERE files into $OUT_DIR"
    python3 - "$DOWNLOAD_DIR/$IMAGING_FILE" "$OUT_DIR" <<'PY'
import re
import shutil
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
manifest = out_dir / "native_unregistered_manifest.txt"

native_modalities = re.compile(
    r"^Imaging/Patient-\d+/week-[^/]+/(CT1|T1|T2|FLAIR)\.nii\.gz$"
)
native_derivatives = re.compile(
    r"^Imaging/Patient-\d+/week-[^/]+/"
    r"(DeepBraTumIA-segmentation/native|HD-GLIO-AUTO-segmentation/native)/"
)

extracted = 0
with zipfile.ZipFile(zip_path) as zf, manifest.open("w", encoding="utf-8") as mf:
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.lstrip("/")
        idx = name.find("Imaging/")
        if idx < 0:
            continue
        rel = name[idx:]
        if not (native_modalities.match(rel) or native_derivatives.match(rel)):
            continue
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, dest.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        mf.write(rel + "\n")
        extracted += 1

if extracted == 0:
    raise SystemExit(
        "No native/unregistered files were extracted. "
        "The Figshare zip layout may have changed."
    )

print(f"Extracted {extracted} native/unregistered files.")
print(f"Manifest: {manifest}")
PY
    ;;
  *)
    echo "Unsupported EXTRACT_MODE=$EXTRACT_MODE; use native_only, full, or none." >&2
    exit 2
    ;;
esac

if [[ "$KEEP_ZIP" != "1" ]]; then
  echo "Removing downloaded imaging zip because KEEP_ZIP=$KEEP_ZIP"
  rm -f "$DOWNLOAD_DIR/$IMAGING_FILE"
fi

echo "Done."
echo "LUMIERE unregistered/native data root: $OUT_DIR"
