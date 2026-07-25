#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh [OPTIONS]

Downloads a pinned 3D Slicer Linux release, installs the extensions used by
MedFlowBench Track A/B, and validates the resulting runtime.

Options:
  --install-dir PATH       Installation directory
  --download-dir PATH      Reusable download cache directory
  --version VERSION        Slicer release (default: 5.12.3)
  --force                  Replace a prior installation made by this script
  --keep-on-failure        Keep a newly extracted installation after failure
  --skip-extensions        Install and validate core Slicer modules only
  --install-system-deps    Install Ubuntu/Debian runtime packages using sudo
  --help                   Show this message

Environment:
  SLICER_INSTALL_DIR               Alternative installation directory
  SLICER_VERSION                   Alternative Slicer release
  SLICER_DOWNLOAD_URL              Override the official download URL
  SLICER_DOWNLOAD_CACHE_DIR        Reusable archive cache directory
  SLICER_SHA512                    Expected archive SHA512
  SLICER_EXTENSIONS_CSV            Required extensions
  SLICER_OPTIONAL_EXTENSIONS_CSV   Best-effort extensions
  SLICER_EXTRA_LIBRARY_PATH        Extra Linux library directory/directories
  XVFB_BIN                         Xvfb executable when DISPLAY is unavailable
  XVFB_LIB_DIR                     Extra library directory for Xvfb
  TABLE1_DISPLAY                   Virtual display number (default: :99)

The default required extension set is:
  WebServer, SegmentEditorExtraEffects, SlicerElastix, QuantitativeReporting

MONAILabel is optional because MedFlowBench communicates with the external
MONAI Label/VISTA3D service directly.
EOF
}

for argument in "$@"; do
  if [[ "$argument" == "-h" || "$argument" == "--help" ]]; then
    usage
    exit 0
  fi
done

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer supports Linux only." >&2
  exit 2
fi
if [[ "$(uname -m)" != "x86_64" && "$(uname -m)" != "amd64" ]]; then
  echo "The official archive used by this script requires x86_64 Linux." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLICER_VERSION="${SLICER_VERSION:-5.12.3}"
INSTALL_DIR="${SLICER_INSTALL_DIR:-}"
DOWNLOAD_URL="${SLICER_DOWNLOAD_URL:-}"
EXPECTED_SHA512="${SLICER_SHA512:-}"
DOWNLOAD_CACHE_DIR="${SLICER_DOWNLOAD_CACHE_DIR:-}"
REQUIRED_EXTENSIONS="${SLICER_EXTENSIONS_CSV:-WebServer,SegmentEditorExtraEffects,SlicerElastix,QuantitativeReporting}"
OPTIONAL_EXTENSIONS="${SLICER_OPTIONAL_EXTENSIONS_CSV:-MONAILabel}"
force=0
keep_on_failure=0
skip_extensions=0
install_system_deps=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR="${2:?--install-dir requires a path}"
      shift 2
      ;;
    --download-dir)
      DOWNLOAD_CACHE_DIR="${2:?--download-dir requires a path}"
      shift 2
      ;;
    --version)
      SLICER_VERSION="${2:?--version requires a value}"
      shift 2
      ;;
    --force)
      force=1
      shift
      ;;
    --keep-on-failure)
      keep_on_failure=1
      shift
      ;;
    --skip-extensions)
      skip_extensions=1
      shift
      ;;
    --install-system-deps)
      install_system_deps=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$INSTALL_DIR" ]]; then
  INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/medflowbench/Slicer-${SLICER_VERSION}"
fi
if [[ -z "$DOWNLOAD_URL" ]]; then
  DOWNLOAD_URL="https://download.slicer.org/download?os=linux&stability=release&version=${SLICER_VERSION}"
fi
if [[ -z "$EXPECTED_SHA512" && "$SLICER_VERSION" == "5.12.3" ]]; then
  EXPECTED_SHA512="66bd3a1b9a7f636b40b96cb8c49f395ee783cdcaf7b43a4b895d6a40df9e0af8393f5ab7631ba50f6bbe06aa17dbcd8a46984a53b693bdf203d34337e2e80401"
fi

for command_name in curl python3 tar; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done
if [[ -n "$EXPECTED_SHA512" ]] && ! command -v sha512sum >/dev/null 2>&1; then
  echo "sha512sum is required when SLICER_SHA512 is set." >&2
  exit 1
fi

install_debian_dependencies() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "--install-system-deps currently supports Ubuntu/Debian only." >&2
    exit 1
  fi
  local sudo_command=()
  if [[ "$(id -u)" != "0" ]]; then
    if ! command -v sudo >/dev/null 2>&1; then
      echo "sudo is required for --install-system-deps when not running as root." >&2
      exit 1
    fi
    sudo_command=(sudo)
  fi
  local audio_package="libasound2"
  if . /etc/os-release 2>/dev/null && [[ "${VERSION_ID:-}" == "24.04" ]]; then
    audio_package="libasound2t64"
  fi
  "${sudo_command[@]}" apt-get update
  "${sudo_command[@]}" apt-get install -y \
    libglu1-mesa libgl1-mesa-dri libglx-mesa0 libegl-mesa0 libgbm1 \
    libpulse-mainloop-glib0 libnss3 "$audio_package" qt5dxcb-plugin libsm6 \
    libxkbcommon0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-render-util0 libxcb-xinerama0 libxkbcommon-x11-0 \
    mesa-utils xvfb x11-utils
}

if [[ "$install_system_deps" == "1" ]]; then
  install_debian_dependencies
fi

INSTALL_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$INSTALL_DIR")"
INSTALL_PARENT="$(dirname "$INSTALL_DIR")"
MARKER_NAME=".medflowbench-slicer-install"
if [[ -e "$INSTALL_DIR" ]]; then
  if [[ "$force" != "1" ]]; then
    echo "Install directory already exists: $INSTALL_DIR" >&2
    echo "Use --force only to replace an installation created by this script." >&2
    exit 1
  fi
  if [[ ! -f "$INSTALL_DIR/$MARKER_NAME" ]]; then
    echo "Refusing to remove an unmarked directory: $INSTALL_DIR" >&2
    exit 1
  fi
  rm -rf "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_PARENT"
work_dir="$(mktemp -d "$INSTALL_PARENT/.slicer-install-${SLICER_VERSION}-XXXXXX")"
archive="$work_dir/Slicer-${SLICER_VERSION}-linux-amd64.tar.gz"
extract_dir="$work_dir/extracted"
created_install=0
xvfb_pid=""

cleanup() {
  local status=$?
  if [[ -n "${download_path:-}" ]]; then
    rm -f "$download_path"
  fi
  if [[ -n "$xvfb_pid" ]] && kill -0 "$xvfb_pid" 2>/dev/null; then
    kill "$xvfb_pid" 2>/dev/null || true
    wait "$xvfb_pid" 2>/dev/null || true
  fi
  rm -rf "$work_dir"
  if [[ "$status" != "0" && "$created_install" == "1" && "$keep_on_failure" != "1" ]]; then
    if [[ -f "$INSTALL_DIR/$MARKER_NAME" ]]; then
      echo "Installation failed; removing the new installation: $INSTALL_DIR" >&2
      rm -rf "$INSTALL_DIR"
    fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "$DOWNLOAD_CACHE_DIR" ]]; then
  mkdir -p "$DOWNLOAD_CACHE_DIR"
  archive="$DOWNLOAD_CACHE_DIR/Slicer-${SLICER_VERSION}-linux-amd64.tar.gz"
fi
archive_is_valid=0
if [[ -f "$archive" && -n "$EXPECTED_SHA512" ]]; then
  cached_sha512="$(sha512sum "$archive" | awk '{print $1}')"
  if [[ "$cached_sha512" == "$EXPECTED_SHA512" ]]; then
    archive_is_valid=1
    echo "[slicer-install] using cached archive: $archive"
  else
    echo "[slicer-install] removing cached archive with an invalid SHA512" >&2
    rm -f "$archive"
  fi
fi
if [[ "$archive_is_valid" != "1" ]]; then
  echo "[slicer-install] downloading $DOWNLOAD_URL"
  download_path="${archive}.partial.$$"
  rm -f "$download_path"
  curl --fail --location --retry 4 --retry-delay 5 --output "$download_path" "$DOWNLOAD_URL"
  mv "$download_path" "$archive"
fi
if [[ -n "$EXPECTED_SHA512" ]]; then
  actual_sha512="$(sha512sum "$archive" | awk '{print $1}')"
  if [[ "$actual_sha512" != "$EXPECTED_SHA512" ]]; then
    echo "Slicer archive SHA512 mismatch." >&2
    echo "Expected: $EXPECTED_SHA512" >&2
    echo "Actual:   $actual_sha512" >&2
    exit 1
  fi
  echo "[slicer-install] SHA512 verified"
else
  echo "[slicer-install] warning: no SHA512 configured for Slicer $SLICER_VERSION" >&2
fi

mkdir -p "$extract_dir"
tar --extract --gzip --file "$archive" --directory "$extract_dir" \
  --no-same-owner --same-permissions

# Some shared filesystems apply a default ACL that strips executable bits
# during extraction even when GNU tar is asked to preserve permissions. Restore
# only files that the signed archive itself marks as executable.
while read -r mode _owner _size _date _time archived_path; do
  if [[ "$mode" == -*x* && -f "$extract_dir/$archived_path" ]]; then
    chmod a+x "$extract_dir/$archived_path"
  fi
done < <(tar --list --verbose --gzip --file "$archive")

mapfile -t extracted_roots < <(find "$extract_dir" -mindepth 1 -maxdepth 1 -type d -print)
if [[ "${#extracted_roots[@]}" != "1" || ! -f "${extracted_roots[0]}/Slicer" ]]; then
  echo "Unexpected Slicer archive layout." >&2
  find "$extract_dir" -mindepth 1 -maxdepth 2 -print >&2
  exit 1
fi
chmod +x "${extracted_roots[0]}/Slicer"
mv "${extracted_roots[0]}" "$INSTALL_DIR"
created_install=1
printf 'version=%s\nsource_url=%s\nsha512=%s\n' \
  "$SLICER_VERSION" "$DOWNLOAD_URL" "$EXPECTED_SHA512" >"$INSTALL_DIR/$MARKER_NAME"

SLICER_EXECUTABLE="$INSTALL_DIR/Slicer"
export SLICER_EXECUTABLE
export SLICER_EXTENSIONS_CSV="$REQUIRED_EXTENSIONS"
export SLICER_OPTIONAL_EXTENSIONS_CSV="$OPTIONAL_EXTENSIONS"
export LANG="${LANG:-C.UTF-8}"

runtime_root="${XDG_RUNTIME_DIR:-${TMPDIR:-/tmp}/medflow-slicer-runtime-${UID:-user}}"
if [[ ! -d "$runtime_root" || ! -w "$runtime_root" ]]; then
  runtime_root="${TMPDIR:-/tmp}/medflow-slicer-runtime-${UID:-user}"
fi
mkdir -p "$runtime_root"
chmod 700 "$runtime_root"
export XDG_RUNTIME_DIR="$runtime_root"

display_is_ready=0
if [[ -n "${DISPLAY:-}" ]]; then
  if ! command -v xdpyinfo >/dev/null 2>&1 || xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    display_is_ready=1
  fi
fi
if [[ "$display_is_ready" != "1" ]]; then
  XVFB_BIN="${XVFB_BIN:-$(command -v Xvfb || true)}"
  if [[ -z "$XVFB_BIN" ]]; then
    echo "No working DISPLAY and Xvfb was not found." >&2
    echo "Install system dependencies or set XVFB_BIN to an existing executable." >&2
    exit 1
  fi
  export DISPLAY="${TABLE1_DISPLAY:-:99}"
  xvfb_log="$work_dir/xvfb.log"
  xvfb_args=("$DISPLAY" -screen 0 1920x1080x24 +extension GLX +render -nolisten tcp)
  if [[ -n "${XVFB_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_xvfb_args <<<"$XVFB_EXTRA_ARGS"
    xvfb_args+=("${extra_xvfb_args[@]}")
  fi
  if [[ -n "${XVFB_LIB_DIR:-}" ]]; then
    env LD_LIBRARY_PATH="$XVFB_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      "$XVFB_BIN" "${xvfb_args[@]}" >"$xvfb_log" 2>&1 &
  else
    "$XVFB_BIN" "${xvfb_args[@]}" >"$xvfb_log" 2>&1 &
  fi
  xvfb_pid=$!
  for _ in $(seq 1 30); do
    if ! kill -0 "$xvfb_pid" 2>/dev/null; then
      break
    fi
    if ! command -v xdpyinfo >/dev/null 2>&1 ||
       xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
      display_is_ready=1
      break
    fi
    sleep 1
  done
  if [[ "$display_is_ready" != "1" ]]; then
    echo "Xvfb failed to start:" >&2
    tail -n 80 "$xvfb_log" >&2 || true
    exit 1
  fi
  echo "[slicer-install] started Xvfb on $DISPLAY"
fi

run_with_slicer_libraries() {
  if [[ -n "${SLICER_EXTRA_LIBRARY_PATH:-}" ]]; then
    env LD_LIBRARY_PATH="$SLICER_EXTRA_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$@"
  else
    "$@"
  fi
}

echo "[slicer-install] validating base application"
run_with_slicer_libraries "$SLICER_EXECUTABLE" \
  --no-splash --no-main-window --testing \
  --python-code "import slicer; print('[slicer-install] version=' + slicer.app.applicationVersion)"

if [[ "$skip_extensions" != "1" ]]; then
  run_with_slicer_libraries bash "$SCRIPT_DIR/install_slicer_extensions.sh"
fi

verification_json="$INSTALL_DIR/medflowbench-slicer-verification.json"
export MEDFLOW_SLICER_VERIFY_OUTPUT="$verification_json"
if [[ "$skip_extensions" == "1" ]]; then
  export SLICER_EXTENSIONS_CSV=""
  export SLICER_OPTIONAL_EXTENSIONS_CSV=""
fi
run_with_slicer_libraries "$SLICER_EXECUTABLE" \
  --no-splash --no-main-window \
  --python-script "$SCRIPT_DIR/verify_slicer_install.py"

echo
echo "3D Slicer installation completed:"
echo "  executable: $SLICER_EXECUTABLE"
echo "  verification: $verification_json"
echo
echo "Set this in .env.table1:"
echo "  SLICER_EXECUTABLE=$SLICER_EXECUTABLE"
