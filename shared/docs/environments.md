# Environment Installation

MedFlowBench uses several processes with different dependency boundaries. A
single environment is not required and is undesirable for reproducibility:
the benchmark host, model server, medical model workers, and desktop
application runtimes can be installed independently.

All repository-managed Python environments require Python 3.10.x. The exact
audited interpreter is Python 3.10.20. Setup and launcher scripts validate the
major/minor version and stop before doing work when another interpreter is
selected.

## Environment Matrix

| Profile | Runs | Python packages | External runtime |
| --- | --- | --- | --- |
| `track_a_b` | Radiology Track A/B host loop | `track_a_b.txt` | 3D Slicer 5.x and required extensions |
| `track_c` | Compressed figures and API Track C | `track_c.txt` | None |
| `pathology` | Pathology benchmark host loop | `pathology.txt` | QuPath 0.7.x and Java 17+ |
| `vllm` | Qwen, Gemma, MedGemma, Lingshu serving | `vllm.txt` | Linux GPU and compatible driver |
| `vista3d` | VISTA3D HTTP segmentation service | `vista3d-server.txt` | Linux CUDA GPU and VISTA3D bundle |
| `mcp` | Optional Slicer MCP endpoint | `mcp.txt` plus provider SDKs | Reachable Slicer WebServer |
| `server` | Optional HTTP tool server | `server` project extra | Reachable Slicer WebServer |
| M3D | Track C M3D worker | Upstream M3D environment | M3D weights and GPU |
| RadFM | Track C RadFM worker | Upstream RadFM environment | RadFM checkout, checkpoint, and GPU |

## Supported Host Environments

From the repository root:

```bash
bash shared/setup/create_python_env.sh track_a_b
source .venv-track_a_b/bin/activate
```

For Track C without 3D Slicer:

```bash
bash shared/setup/create_python_env.sh track_c
source .venv-track_c/bin/activate
```

API-backed open-weight Track C runs also need the separate vLLM environment.
After filling `shared/config/track_c.env.example`, the launcher starts and
stops that server automatically:

```bash
bash radiology/model_serving/create_vllm_env.sh .venv-vllm
bash shared/setup/run_release_checks.sh --scope radiology
cp shared/config/track_c.env.example .env.track_c
bash radiology/track_c/scripts/reproduce_track_c.sh
```

No 3D Slicer, Xvfb, GLX, WebServer, or extension dependency applies to Track C.

For pathology with QuPath:

```bash
bash shared/setup/create_python_env.sh pathology
source .venv-pathology/bin/activate
export QUPATH_EXECUTABLE=/absolute/path/to/QuPath
```

The QuPath direct-control path also needs a Java 17+ JDK. The optional
Midscene sidecar is a separate Node.js environment:

```bash
cd pathology/runtime/qupath_sidecar
npm ci
npx --no-install tsx src/cli.ts env-check
```

## 3D Slicer

3D Slicer is an application and is not installed by pip. On x86_64 Linux the
repository can install the pinned application and extensions together:

```bash
bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh
```

The installer defaults to Slicer 5.12.3, verifies its published SHA512,
installs into a writable user directory, and rolls back a new installation
when extension or module validation fails. Use `--install-dir PATH` to choose
another location and `--download-dir PATH` for a reusable archive cache. On
Ubuntu/Debian, `--install-system-deps` installs the
documented X11, Mesa/GLX, and runtime libraries through sudo. The final
verification creates a Qt OpenGL context and fails if the selected display
cannot support Slicer rendering. The installer also restores executable bits
recorded in the signed archive, which is needed on some shared filesystems whose
default ACL changes extraction permissions. For a pre-existing installation,
verify `test -x "$SLICER_EXECUTABLE"`; reinstall from the pinned archive instead
of applying recursive `chmod` when permissions are incomplete.

To use a pre-existing Slicer 5.x build, install only the benchmark extensions:

```bash
export SLICER_EXECUTABLE=/absolute/path/to/Slicer
bash radiology/slicer_track_a_b/setup/install_slicer_extensions.sh
```

On Debian/Ubuntu systems without sudo, the repository can extract the required
Xvfb, Mesa, XKB, and Qt packages into a user-owned directory:

```bash
bash radiology/slicer_track_a_b/setup/install_slicer_user_deps.sh \
  --install-dir /absolute/path/to/slicer-user-deps
source /absolute/path/to/slicer-user-deps/medflowbench-slicer-user-deps.env
bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh \
  --install-dir /absolute/path/to/Slicer-5.12.3
```

This helper uses the host's configured apt repositories and never modifies the
system package database. It currently supports Debian/Ubuntu amd64 hosts.

Track A/B needs WebServer, which is bundled with the pinned Slicer 5.12.3
release. For older releases, the extension installer obtains WebServer from
the matching Extension Manager catalog when necessary. Toolpack experiments
additionally use SegmentEditorExtraEffects, BRAINSFit/resampling modules
distributed with Slicer, SlicerElastix for the Elastix path, and
QuantitativeReporting for DICOM SEG export. The external MONAI Label/VISTA3D
service does not require the Slicer MONAILabel extension. Headless Linux runs
also need Xvfb and a valid `DISPLAY`.
On Ubuntu/Debian, install `xvfb`, `x11-xkb-utils`, `x11-utils`, and
`libxkbcommon0`; installing only the `Xvfb` binary can leave `xkbcomp` or the
Qt XKB runtime unavailable. User-space installations can be supplied through
the `XVFB_BIN`, `XVFB_LIB_DIR`, and `SLICER_EXTRA_LIBRARY_PATH` fields in
`.env.table1`.

## Open-Weight Models Through vLLM

The vLLM server is independent of the Track A/B or Track C host:

```bash
bash radiology/model_serving/create_vllm_env.sh .venv-vllm
```

The bootstrap Python must include `venv` and `ensurepip`. Set
`PYTHON_BOOTSTRAP=/path/to/conda/env/bin/python` on managed systems where the
system Python omits them. Slurm runs automatically use a node-local vLLM cache;
`MEDFLOW_VLLM_CACHE_DIR` provides the same behavior on other schedulers. The
serving wrapper falls back to the PyTorch sampler when `nvcc` or `ninja` is
unavailable.

By default the installer consumes
`shared/requirements/locks/vllm-linux-py310-cu13-20260725.txt`, exported from
the successful clean MedGemma reproduction. It contains every resolved Python
package, including Torch 2.11.0, Transformers 4.57.6, vLLM 0.21.0, and pytest
8.4.2. Set `VLLM_USE_LOCK=0` only when a different accelerator platform
requires vLLM to resolve another wheel stack from the shorter curated
requirements. That fallback is supported but is not the exact audited lock.

Serving commands and model IDs are documented in `open_weight_models.md`.
Keeping vLLM separate prevents its CUDA-specific Torch stack from changing the
benchmark host environment.

## VISTA3D

The public setup profile reproduces the audited server's Python 3.10.20,
PyTorch 2.5.1/CUDA 12.1, Torchvision 0.20.1, MONAI 1.5.2, and MONAI Label
0.8.5 combination:

```bash
bash shared/setup/create_python_env.sh vista3d
source .venv-vista3d/bin/activate
python -m monai.bundle download vista3d --bundle_dir /path/to/monai-bundles
python radiology/slicer_track_a_b/services/vista3d_label_app.py \
  --bundle_dir /path/to/monai-bundles/vista3d \
  --work_dir ./runs/vista3d-server \
  --host 127.0.0.1 \
  --port 8001
```

Set `VISTA3D_TORCH_INDEX_URL` and the two version variables accepted by the
setup script when the machine requires a different CUDA/PyTorch build. The
CUDA driver is a system prerequisite and is not installed by this repository.

## M3D And RadFM

M3D and RadFM use their upstream model code and pinned ML stacks. Install each
in its own environment, then pass that environment's interpreter to the
benchmark:

```bash
--method m3d \
--m3d_python /path/to/m3d-env/bin/python \
--m3d_model_name_or_path /path/to/M3D-weights
```

```bash
--method radfm \
--radfm_python /path/to/radfm-env/bin/python \
--radfm_repo_root /path/to/RadFM \
--radfm_checkpoint /path/to/checkpoint
```

The included workers define the input normalization and inference adapter, but
the repository does not vendor upstream model code or redistribute weights.

## Audited Server Environments

The sanitized Track A/B host, Track C host, and VISTA3D service snapshots used
Conda-prefix environments with Python 3.10.20. On inspection, the first two
environments were nearly identical and contained
vLLM, Torch, Transformers, MCP, and HTTP-server dependencies. This reflects
development-time environment reuse, not a code requirement. Sanitized package
inventories are retained under `shared/requirements/snapshots/`; use the
curated profiles above for a clean reproduction.
