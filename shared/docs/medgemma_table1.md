# Reproduce the MedGemma Table 1 Rows

This procedure runs the two released MedGemma configurations through the
viewer-native radiology Track A protocol:

| Manifest key | Model source | Serving path |
| --- | --- | --- |
| `medgemma-1.5-4b-it` | `google/medgemma-1.5-4b-it` | Local vLLM OpenAI-compatible server |
| `medgemma-27b-it` | `google/medgemma-27b-it` | Local vLLM OpenAI-compatible server |

The benchmark host does not call a closed-source model for these rows.
MedGemma weights are gated: accept the upstream terms using the same Hugging
Face account represented by `HF_TOKEN` before starting vLLM.

This document first covers viewer-native Track A. The independent Track C
one-command path is documented in the final section and does not require
3D Slicer.

## 1. Prepare Data and 3D Slicer

Download the three radiology datasets as described in
`shared/data_download/README.md`. On x86_64 Linux, install the pinned 3D
Slicer build and all required extensions:

```bash
bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh
```

The script prints the resulting `SLICER_EXECUTABLE` value. To use an existing
Slicer build instead, set that variable and run
`install_slicer_extensions.sh`. The pinned Slicer 5.12.3 package already
contains WebServer; the installer adds SegmentEditorExtraEffects,
SlicerElastix, and QuantitativeReporting, then verifies their benchmark-facing
features. MONAILabel is installed on a best-effort basis because the benchmark
uses the external MONAI Label/VISTA3D HTTP service.

On Ubuntu/Debian headless hosts, the installer can install the complete system
runtime when sudo is available:

```bash
bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh \
  --install-system-deps
```

On Debian/Ubuntu clusters without sudo, install the same runtime packages into
a private directory, source the generated environment file, and then install
Slicer:

```bash
bash radiology/slicer_track_a_b/setup/install_slicer_user_deps.sh \
  --install-dir /absolute/path/to/slicer-user-deps
source /absolute/path/to/slicer-user-deps/medflowbench-slicer-user-deps.env
bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh \
  --install-dir /absolute/path/to/Slicer-5.12.3 \
  --download-dir /absolute/path/to/download-cache
```

The user-space helper uses `apt-get --download-only` and `dpkg-deb -x`; it
does not run `sudo`, change the system package database, or redistribute the
packages.

The Table 1 launcher starts Xvfb when `DISPLAY` is unset. On clusters without
administrator access, point `XVFB_BIN`, `XVFB_LIB_DIR`, and
`SLICER_EXTRA_LIBRARY_PATH` in `.env.table1` at equivalent user-installed
packages. The Xvfb build must advertise GLX in `xdpyinfo`; merely accepting X11
connections is insufficient for Slicer slice rendering.

## 2. Create the vLLM Environment

Use a Linux GPU host and Python 3.10.x. The clean reproduction used Python
3.10.20. The installer rejects other Python major/minor versions:

```bash
bash radiology/model_serving/create_vllm_env.sh .venv-vllm
bash shared/setup/run_release_checks.sh --scope radiology
export HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
```

If the system Python reports that `ensurepip` is unavailable, install
`python3-venv`, load a Python module, or use a Conda Python only as the
bootstrap executable:

```bash
PYTHON_BOOTSTRAP=/path/to/conda/env/bin/python \
  bash radiology/model_serving/create_vllm_env.sh .venv-vllm
```

The environment creator uses an isolated pip cache below `.venv-vllm` by
default and retries transient downloads. Override `PIP_CACHE_DIR`,
`PIP_INSTALL_ATTEMPTS`, `PIP_NETWORK_RETRIES`, or `PIP_NETWORK_TIMEOUT` when
required by a local package mirror.

The default install consumes the exact package lock from the audited
Linux/Python 3.10.20/CUDA 13 run. It includes vLLM 0.21.0, Torch 2.11.0,
Transformers 4.57.6, and pytest 8.4.2. Use `VLLM_USE_LOCK=0` only on an
accelerator platform that cannot use this wheel stack.

Model size, precision, context length, and quantization determine the required
GPU count and memory. Configure the deployment rather than relying on a
hard-coded hardware assumption:

```bash
export VLLM_TENSOR_PARALLEL_SIZE=1
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_MAX_MODEL_LEN=32768
export LOCAL_VLLM_MAX_TOOL_CALLS_PER_TURN=1
export LOCAL_VLLM_MAX_INPUT_IMAGES=16
export LOCAL_VLLM_MAX_HISTORY_TOOL_ROUNDS=3
export LOCAL_VLLM_MAX_TOOL_RESULT_CHARS=8000
export LOCAL_VLLM_MAX_OUTPUT_TOKENS=2048
export VLLM_NO_USAGE_STATS=1
```

On Slurm, the serving script automatically places compile caches under the
job's node-local `TMPDIR`. Elsewhere, set `MEDFLOW_VLLM_CACHE_DIR` explicitly
when the home quota is small. If `nvcc` or `ninja` is absent, the launcher
selects the vLLM PyTorch sampler automatically; installing a full CUDA build
toolchain is not required merely to serve MedGemma.

`LOCAL_VLLM_MAX_OUTPUT_TOKENS` caps each agent turn. Keep the documented
default unless a model's structured final answer is demonstrably truncated.
The local manual tool protocol executes one tool per turn, retains the original
task plus the latest three tool rounds, and sends at most 16 images. The live
Slicer scene persists when older prompt messages are removed. These bounds
prevent repeated viewer actions from overflowing MedGemma's 32K context.
`VLLM_NO_USAGE_STATS=1` prevents telemetry writes to quota-limited home
directories.

Use `VLLM_MODEL_PATH_MEDGEMMA_1_5_4B_IT` and
`VLLM_MODEL_PATH_MEDGEMMA_27B_IT` in the environment file when weights are
already available locally.

## 3. Configure the Experiment

Create the untracked configuration:

```bash
cp shared/config/table1.env.example .env.table1
```

Set at least:

```text
PYTHON_BIN=/absolute/path/to/repository/.venv-vllm/bin/python
VLLM_BIN=/absolute/path/to/repository/.venv-vllm/bin/vllm
SLICER_EXECUTABLE=/absolute/path/to/Slicer
UCSF_DATA_ROOT=...
UCSF_METADATA_CSV=...
NSCLC_DATA_ROOT=...
NSCLC_METADATA_CSV=...
NSCLC_LABELS_CSV=...
LUMIERE_DATA_ROOT=...
LUMIERE_RATINGS_CSV=...
LUMIERE_COMPLETENESS_CSV=...
TABLE1_RESULTS_ROOT=/absolute/path/to/results/medgemma-table1
HF_TOKEN=...
```

No OpenAI, Gemini, or Anthropic credential is required when
`--model-set medgemma` is selected.

## 4. Run the Five-Case Reproduction

The default is five cases/questions per dataset. Run one MedGemma model:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --models medgemma-1.5-4b-it
```

The launcher validates the manifest, starts one MedGemma vLLM server at a
time, checks its health and model endpoint, runs five units from each dataset,
and writes the same result schema used by the full experiment.

## 5. Run the Full Table 1 Protocol

Pass `--limit 0` for the full eligible cohorts:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --model-set medgemma \
  --limit 0
```

The released protocol uses registered LUMIERE inputs, 20 agent rounds, and no
Track B toolpacks. The launcher expects 139 LUMIERE questions, 495 UCSF-PDGM
cases, and 162 NSCLC cases, resumes completed model/dataset pairs, and writes:

```text
TABLE1_RESULTS_ROOT/
  medgemma-1.5-4b-it/
  medgemma-27b-it/
  table1_results.csv
  table1_results.json
```

Per-case directories contain the rendered evidence, model output, requested
model identifier, prompt provenance, tool trace, and deterministic scores.

## Run Only One MedGemma Model

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --models medgemma-1.5-4b-it \
  --limit 0
```

or:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --models medgemma-27b-it \
  --limit 0
```

If a model cannot fit with the default serving configuration, adjust tensor
parallelism, maximum context length, quantization, or the model path through
the documented vLLM environment variables. Those deployment changes do not
change the benchmark prompts, dataset protocol, scoring, or output schema.

## Track C With MedGemma

Track C uses the same model manifest and vLLM serving path but no viewer
runtime. Create `.env.track_c` from its dedicated template and set the same
dataset/model paths:

```bash
cp shared/config/track_c.env.example .env.track_c
bash radiology/track_c/scripts/reproduce_track_c.sh
```

The default invocation runs `medgemma-1.5-4b-it` with the paper's
`compressed_figures` representation and `limit=5`. Use `--limit 0` for the
full eligible cohorts or `--models medgemma-1.5-4b-it,medgemma-27b-it` for
both MedGemma rows. Track C does not start Slicer, WebServer, Xvfb, or any
Slicer extension.

## Expected Warnings

- Some UCSF-PDGM NIfTI headers contain a nonstandard `pixdim[0]` value.
  Nibabel reports that it is using the NIfTI-defined default `qfac=1`. This is
  a source-data header warning; the runner does not modify the input file.
- vLLM 0.21 reports that its Transformers v4 compatibility path is
  deprecated. The audited lock intentionally keeps Transformers 4.57.6, which
  completed both Track A and Track C. Upgrade that stack only after validating
  the selected open-weight models with a new lock.
- `nvcc or ninja not found; using the PyTorch sampler` is an intentional
  fallback on runtime-only GPU hosts, not a failed model launch. A successful
  launch prints `vLLM Responses API ready`.
