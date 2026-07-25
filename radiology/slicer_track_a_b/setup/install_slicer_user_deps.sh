#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash radiology/slicer_track_a_b/setup/install_slicer_user_deps.sh [OPTIONS]

Downloads and extracts the Debian/Ubuntu Xvfb, Mesa, XKB, and Qt runtime
packages needed by a headless 3D Slicer installation without using sudo.
No system package is installed or modified. A private mount-namespace wrapper
maps xkbcomp to the absolute path expected by the Ubuntu X server.

Options:
  --install-dir PATH  User-space dependency directory
  --force             Replace a directory created by this script
  --help              Show this message

Default install directory:
  REPO_ROOT/.runtime/slicer-user-deps
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
INSTALL_DIR="$REPO_ROOT/.runtime/slicer-user-deps"
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-dir)
      INSTALL_DIR="${2:?--install-dir requires a path}"
      shift 2
      ;;
    --force)
      force=1
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

if [[ "$(uname -s)" != "Linux" || ! -f /etc/debian_version ]]; then
  echo "User-space package extraction currently supports Debian/Ubuntu Linux only." >&2
  exit 1
fi
if [[ "$(dpkg --print-architecture)" != "amd64" ]]; then
  echo "The released 3D Slicer Linux build and this helper require amd64." >&2
  exit 1
fi
for command_name in apt-get dpkg dpkg-deb; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command not found: $command_name" >&2
    exit 1
  fi
done

INSTALL_DIR="$(python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(sys.argv[1])))' "$INSTALL_DIR")"
MARKER="$INSTALL_DIR/.medflowbench-slicer-user-deps"
if [[ -e "$INSTALL_DIR" ]]; then
  if [[ "$force" != "1" ]]; then
    echo "Install directory already exists: $INSTALL_DIR" >&2
    echo "Use --force only to replace a directory created by this script." >&2
    exit 1
  fi
  if [[ ! -f "$MARKER" ]]; then
    echo "Refusing to remove an unmarked directory: $INSTALL_DIR" >&2
    exit 1
  fi
  rm -rf "$INSTALL_DIR"
fi

PACKAGE_DIR="$INSTALL_DIR/packages"
ROOT_DIR="$INSTALL_DIR/root"
mkdir -p "$PACKAGE_DIR" "$ROOT_DIR"
touch "$MARKER"
created_install=1

cleanup() {
  local status=$?
  if [[ "$status" != "0" && "$created_install" == "1" && -f "$MARKER" ]]; then
    echo "Dependency installation failed; removing $INSTALL_DIR" >&2
    rm -rf "$INSTALL_DIR"
  fi
  trap - EXIT
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

packages=(
  xvfb
  xserver-common
  x11-xkb-utils
  x11-utils
  xauth
  xfonts-base
  xkb-data
  libglu1-mesa
  libgl1-mesa-dri
  libglx-mesa0
  libegl-mesa0
  libgbm1
  libpulse-mainloop-glib0
  libnss3
  libasound2
  libsm6
  libxkbcommon0
  libxcb-icccm4
  libxcb-image0
  libxcb-keysyms1
  libxcb-render-util0
  libxcb-xinerama0
  libxkbcommon-x11-0
  bubblewrap
)

mkdir -p "$PACKAGE_DIR/partial"
echo "Resolving and downloading the missing user-space runtime packages."
apt-get \
  --download-only \
  --no-install-recommends \
  --yes \
  -o Debug::NoLocking=true \
  -o Acquire::Retries=5 \
  -o Acquire::http::Timeout=120 \
  -o Acquire::https::Timeout=120 \
  -o APT::Keep-Downloaded-Packages=true \
  -o Dir::Cache::archives="$PACKAGE_DIR" \
  install "${packages[@]}"

mapfile -d '' -t deb_files < <(
  find "$PACKAGE_DIR" -maxdepth 1 -type f -name '*.deb' -print0
)
if [[ "${#deb_files[@]}" == "0" ]]; then
  echo "APT completed but no package archives were found in $PACKAGE_DIR." >&2
  exit 1
fi
echo "Extracting ${#deb_files[@]} downloaded package archives."
for deb_file in "${deb_files[@]}"; do
  dpkg-deb -x "$deb_file" "$ROOT_DIR"
done

XVFB_REAL_BIN="$ROOT_DIR/usr/bin/Xvfb"
XKBCOMP_BIN="$ROOT_DIR/usr/bin/xkbcomp"
BWRAP_BIN="$ROOT_DIR/usr/bin/bwrap"
if [[ ! -f "$XVFB_REAL_BIN" || ! -f "$XKBCOMP_BIN" || ! -f "$BWRAP_BIN" ]]; then
  echo "The extracted package set did not provide Xvfb, xkbcomp, and bubblewrap." >&2
  exit 1
fi
# Ubuntu merges /bin into /usr/bin. The private /usr/bin bind below therefore
# also needs a shell for the command that Xvfb uses to invoke xkbcomp.
cp -L /bin/sh "$ROOT_DIR/usr/bin/sh"
# Some shared filesystems apply a default ACL that strips archive executable
# bits. Restore only the package-provided entry points used by this helper.
chmod a+x "$XVFB_REAL_BIN" "$XKBCOMP_BIN" "$BWRAP_BIN" "$ROOT_DIR/usr/bin/sh"

library_dirs=()
for candidate in \
  "$ROOT_DIR/usr/lib/x86_64-linux-gnu" \
  "$ROOT_DIR/usr/lib/x86_64-linux-gnu/pulseaudio" \
  "$ROOT_DIR/lib/x86_64-linux-gnu" \
  "$ROOT_DIR/usr/lib"; do
  if [[ -d "$candidate" ]]; then
    library_dirs+=("$candidate")
  fi
done
XVFB_LIB_DIR="$(IFS=:; echo "${library_dirs[*]}")"
driver_dirs=()
for candidate in \
  "$ROOT_DIR/usr/lib/x86_64-linux-gnu/dri" \
  "$ROOT_DIR/lib/x86_64-linux-gnu/dri" \
  /usr/lib/x86_64-linux-gnu/dri \
  /lib/x86_64-linux-gnu/dri; do
  if [[ -d "$candidate" ]]; then
    driver_dirs+=("$candidate")
  fi
done
if [[ "${#driver_dirs[@]}" == "0" ]]; then
  echo "Mesa was requested, but no DRI driver directory is available." >&2
  exit 1
fi
LIBGL_DRIVERS_PATH_VALUE="$(IFS=:; echo "${driver_dirs[*]}")"
for executable in "$XVFB_REAL_BIN" "$BWRAP_BIN"; do
  if LD_LIBRARY_PATH="$XVFB_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
     ldd "$executable" | grep -q "not found"; then
    echo "The extracted executable still has unresolved shared libraries: $executable" >&2
    LD_LIBRARY_PATH="$XVFB_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      ldd "$executable" >&2
    exit 1
  fi
done

mkdir -p "$INSTALL_DIR/bin"
XVFB_BIN="$INSTALL_DIR/bin/Xvfb"
cat >"$XVFB_BIN" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="$XVFB_LIB_DIR\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$BWRAP_BIN" \
  --die-with-parent \
  --ro-bind / / \
  --dev-bind /dev /dev \
  --bind /tmp /tmp \
  --ro-bind "$ROOT_DIR/usr/bin" /usr/bin \
  -- "$XVFB_REAL_BIN" "\$@"
EOF
chmod a+x "$XVFB_BIN"
if [[ ! -x "$XVFB_BIN" ]]; then
  echo "Could not create the user-space Xvfb wrapper: $XVFB_BIN" >&2
  exit 1
fi

ENV_FILE="$INSTALL_DIR/medflowbench-slicer-user-deps.env"
cat >"$ENV_FILE" <<EOF
export PATH="$ROOT_DIR/usr/bin:\${PATH}"
export XVFB_BIN="$XVFB_BIN"
export XVFB_LIB_DIR="$XVFB_LIB_DIR"
export SLICER_EXTRA_LIBRARY_PATH="$XVFB_LIB_DIR"
export LIBGL_DRIVERS_PATH="$LIBGL_DRIVERS_PATH_VALUE\${LIBGL_DRIVERS_PATH:+:\$LIBGL_DRIVERS_PATH}"
EOF

cat <<EOF
User-space Slicer runtime dependencies installed:
  root:     $ROOT_DIR
  env file: $ENV_FILE

Load them before installing or running Slicer:
  source "$ENV_FILE"
EOF

created_install=0
trap - EXIT
