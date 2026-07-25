---
name: reproduce-medflowbench-radiology
description: Prepare, run, debug, resume, and validate MedFlowBench radiology experiments from a clean checkout. Use when an agent must reproduce viewer-native Track A Table 1 with 3D Slicer, run Track C compressed-figure or sampled-slice baselines without Slicer, serve MedGemma/Qwen/Gemma/Lingshu through vLLM, download or locate the public datasets, or verify that a limit-5 or full run is release-ready.
---

# Reproduce MedFlowBench Radiology

Run this workflow from the repository root. Discover or request missing
machine-specific paths; never invent them.

## Choose The Requested Scope

| Scope | Primary command | Slicer required |
| --- | --- | --- |
| Track A smoke | Table 1 launcher, one model, `--limit 5` | Yes |
| Track A full | Table 1 launcher, selected model set, `--limit 0` | Yes |
| Track C smoke | Track C launcher, one model, `--limit 5` | No |
| Track C full | Track C launcher, selected model set, `--limit 0` | No |
| Radiology release check | Track A smoke followed by Track C smoke | Only for Track A |

Track A is the viewer-native Table 1 protocol. Track B uses the same Slicer
runners with optional segmentation, registration, MONAI Label, or VISTA3D
toolpacks. Track C is separate Python code that creates image/volume artifacts
and never starts or connects to Slicer.

When the request says only "make the release reproducible", run both Track A
and Track C limit-5 smoke workflows. Do not treat a successful Track A run as
proof that Track C works, or vice versa.

Pathology with QuPath is outside this skill. Use
`shared/docs/pathology.md`.

## Non-Negotiable Rules

1. Use Linux x86_64 for local vLLM and Track A.
2. Use Python 3.10.x. The audited interpreter is Python 3.10.20.
3. Keep data, weights, environments, credentials, downloads, and results
   outside the release tree.
4. Never commit `.env` files, credentials, patient data, weights, runs, logs,
   screenshots, or generated evidence.
5. Preserve source data and existing installations. Delete or replace only
   paths created for the current clean-room test.
6. Run downloads and GPU work in `screen`, `tmux`, or equivalent persistent
   execution.
7. Bind vLLM and Slicer WebServer to localhost or a protected cluster network.
8. Never use Slicer's `--no-main-window`; Track A needs the Qt rendering layout.
9. Never start Slicer, Xvfb, or WebServer for Track C.
10. Do not edit prompts, scoring, cohort filters, expected counts, or model
    manifests to make a run pass.
11. Validate `n_failed`, eligible counts, completion markers, aggregate files,
    and logs. Process exit alone is insufficient.

## Step 1: Establish Paths And Persistent Execution

Resolve these absolute paths before installing or running anything:

```text
REPO_ROOT
DATA_ROOT
RESULTS_ROOT
RUNTIME_ROOT
DOWNLOAD_CACHE
PYTHON_BOOTSTRAP
MODEL_PATH or Hugging Face access
SLICER_EXECUTABLE          Track A only
```

Use project or scratch storage for large files and caches. On a remote host:

```bash
screen -S medflow-reproduction
```

Enter an existing Slurm allocation from inside the persistent shell using the
site-documented `srun` command. Do not request another GPU when an allocation
was provided. Record the job ID and compute hostname.

Confirm the checkout:

```bash
test -f pyproject.toml
test -f radiology/slicer_track_a_b/scripts/reproduce_table1.sh
test -f radiology/track_c/scripts/reproduce_track_c.sh
test -f shared/config/table1.env.example
test -f shared/config/track_c.env.example
```

## Step 2: Preflight The Host

Run:

```bash
uname -s
uname -m
nvidia-smi
df -h "$DATA_ROOT" "$RESULTS_ROOT" "$RUNTIME_ROOT"
command -v curl
command -v java || true
command -v python3.10 || true
ulimit -n
ss -ltn 2>/dev/null | grep -E ':(8000|2016) ' || true
```

Require Linux `x86_64`/`amd64` and a vLLM-supported GPU for local open-weight
models. Java is needed by the NBIA downloader. Track A additionally needs
OpenGL and either system Xvfb/Mesa or the included user-space dependencies.

Port 8000 is the default vLLM port. Port 2016 is the default Slicer WebServer
port and is irrelevant to Track C. Reuse an occupied service only when it is a
known compatible process; otherwise select an unused port in the environment
file.

## Step 3: Obtain And Validate The Data

Both tracks use the same UCSF-PDGM, NSCLC-Radiogenomics, and registered
LUMIERE cohorts.

Check public endpoints before downloading imaging:

```bash
DATA_ROOT=/absolute/path/to/data \
WORK_ROOT=/absolute/path/to/download-work \
  bash shared/data_download/download_tcia_radiology_collections.sh smoke
```

Start or resume the public downloads:

```bash
DATA_ROOT=/absolute/path/to/data \
WORK_ROOT=/absolute/path/to/download-work \
  bash shared/data_download/download_tcia_radiology_collections.sh resume
```

The downloader preserves upstream assets:

- NSCLC is stored below `DATA_ROOT/nsclc-radiogenomics`.
- UCSF assets are staged below `DATA_ROOT/ucsf-pdgm/.flat`; inspect archive
  types and extract them without flattening case directories.
- LUMIERE files are stored below `DATA_ROOT/LUMIERE`.

Build the registered LUMIERE tree used by Track A Table 1 and Track C:

```bash
OUT_DIR=/absolute/path/to/data/LUMIERE \
EXTRACT_MODE=full \
  bash shared/data_download/download_lumiere_unregistered.sh
```

Use a separate `EXTRACT_MODE=native_only` destination only for later Track B
registration experiments.

Locate metadata instead of guessing:

```bash
find "$DATA_ROOT" -type f \
  \( -name 'UCSF-PDGM-metadata_v5.csv' \
     -o -name 'metadata.csv' \
     -o -name 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv' \
     -o -name 'LUMIERE-ExpertRating-v202211.csv' \
     -o -name 'LUMIERE-datacompleteness.csv' \) -print
```

Verify the logical layout:

```text
UCSF_DATA_ROOT/
  UCSF-PDGM-*_nifti/
  UCSF-PDGM-metadata_v5.csv
NSCLC_DATA_ROOT/
  DICOM collection files
  metadata.csv
  NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv
LUMIERE_DATA_ROOT/
  Imaging/Patient-*/week-*/
  LUMIERE-ExpertRating-v202211.csv
  LUMIERE-datacompleteness.csv
```

The code license is not a dataset license. Preserve upstream terms.

## Step 4: Create The Python And vLLM Environment

Check the bootstrap:

```bash
/absolute/path/to/python3.10 --version
/absolute/path/to/python3.10 -c 'import ensurepip; print("ensurepip OK")'
```

For MedGemma, Qwen, Gemma, or Lingshu in either track, create the audited
combined host/vLLM environment:

```bash
PYTHON_BOOTSTRAP=/absolute/path/to/python3.10 \
  bash radiology/model_serving/create_vllm_env.sh \
  /absolute/path/to/runtime/.venv-vllm
```

The default lock is
`shared/requirements/locks/vllm-linux-py310-cu13-20260725.txt`. Audited core
versions are Python 3.10.20, vLLM 0.21.0, PyTorch 2.11.0, Transformers 4.57.6,
and pytest 8.4.2.

vLLM 0.21 may print a future deprecation warning for its Transformers v4
codepath. This is expected with the audited lock; do not upgrade the lock
during a reproduction run solely to remove that warning.

Use `VLLM_USE_LOCK=0` only when the platform cannot install the audited CUDA
stack. Record the replacement environment as a portability run.

For hosted-only Track C without local vLLM, this smaller environment is valid:

```bash
PYTHON_BOOTSTRAP=/absolute/path/to/python3.10 \
  bash shared/setup/create_python_env.sh track_c \
  /absolute/path/to/runtime/.venv-track_c
```

Run source checks with the selected Python:

```bash
PYTHON_BIN=/absolute/path/to/runtime/.venv-vllm/bin/python \
  bash shared/setup/run_release_checks.sh --scope radiology
```

Do not continue unless `pip check`, syntax checks, runner import/help checks,
and unit tests pass.

## Step 5: Configure Models

`shared/config/table1_models.json` is authoritative for hosted and open-weight
model keys. It includes Qwen3.5, Gemma 3, MedGemma, and Lingshu through the
same local OpenAI-compatible vLLM path.

For gated Gemma/MedGemma repositories, accept the upstream terms and provide
`HF_TOKEN` through the shell or an untracked environment file. Alternatively,
set the matching `VLLM_MODEL_PATH_*` variable to an existing local weight
directory.

Keep the tested local context controls unless deliberately testing a new
deployment:

```text
VLLM_IMAGE_LIMIT=16
LOCAL_VLLM_MAX_TOOL_CALLS_PER_TURN=1
LOCAL_VLLM_MAX_INPUT_IMAGES=16
LOCAL_VLLM_MAX_HISTORY_TOOL_ROUNDS=3
LOCAL_VLLM_MAX_TOOL_RESULT_CHARS=8000
LOCAL_VLLM_MAX_OUTPUT_TOKENS=2048
VLLM_NO_USAGE_STATS=1
```

Set `MEDFLOW_VLLM_CACHE_DIR` to scratch storage on quota-limited clusters.
Do not start vLLM manually for the standard launchers; they start,
health-check, and stop one local model server at a time.

Hosted rows require only their selected credentials:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`/`GOOGLE_API_KEY`, or Vertex AI credentials
- `ANTHROPIC_API_KEY`, or Vertex AI credentials

## Step 6A: Prepare Track A

Skip this section for Track C-only work.

Install the pinned Slicer 5.12.3 Linux runtime:

```bash
bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh \
  --install-dir /absolute/path/to/runtime/Slicer-5.12.3 \
  --download-dir /absolute/path/to/download-cache \
  --install-system-deps
```

Without sudo on Debian/Ubuntu:

```bash
bash radiology/slicer_track_a_b/setup/install_slicer_user_deps.sh \
  --install-dir /absolute/path/to/runtime/slicer-user-deps

source /absolute/path/to/runtime/slicer-user-deps/medflowbench-slicer-user-deps.env

bash radiology/slicer_track_a_b/setup/install_slicer_linux.sh \
  --install-dir /absolute/path/to/runtime/Slicer-5.12.3 \
  --download-dir /absolute/path/to/download-cache
```

For an existing Slicer 5.x installation:

```bash
export SLICER_EXECUTABLE=/absolute/path/to/Slicer
bash radiology/slicer_track_a_b/setup/install_slicer_extensions.sh
```

Require OpenGL, BRAINSFit, BRAINSResample, WebServer,
SegmentEditorExtraEffects/Local Threshold, SlicerElastix, and
QuantitativeReporting/DICOM SEG. The verification JSON must contain
`"errors": []`. MONAILabel is optional for Table 1.

Create Track A configuration:

```bash
cp shared/config/table1.env.example .env.table1
chmod 600 .env.table1
```

Fill all absolute Python, Slicer, dataset, metadata, result, vLLM, and selected
model paths. Check placeholders without printing credentials:

```bash
grep -nE '^[A-Z0-9_]+=/path/to' .env.table1
```

## Step 6C: Prepare Track C

Track C must not receive `SLICER_EXECUTABLE`, `DISPLAY`, WebServer, Xvfb, or
extension configuration.

Create its configuration:

```bash
cp shared/config/track_c.env.example .env.track_c
chmod 600 .env.track_c
```

Fill `PYTHON_BIN`, all three dataset roots and metadata files,
`TRACK_C_RESULTS_ROOT`, `VLLM_BIN`, and selected model paths or credentials.
The same `.venv-vllm`, data, and local weights may be referenced by both
tracks.

```bash
grep -nE '^[A-Z0-9_]+=/path/to' .env.track_c
```

The paper's 2D montage baseline is `compressed_figures`. `table3_api` is the
sampled-slice API method. M3D and RadFM require incompatible upstream worker
environments and are run directly as documented in `shared/docs/track_c.md`.

## Step 7: Run Limit-5 Smoke Tests

Track A:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --models medgemma-1.5-4b-it \
  --limit 5 \
  --no-resume
```

This runs registered LUMIERE, UCSF-PDGM, and NSCLC with 20 agent rounds and no
Track B optional toolpacks.

Track C:

```bash
bash radiology/track_c/scripts/reproduce_track_c.sh \
  --model medgemma-1.5-4b-it \
  --method compressed_figures \
  --limit 5 \
  --no-resume
```

This runs the same three cohorts without Slicer and writes deterministic
Track C scores. For a general release check, both commands must pass.

Do not proceed to full cohorts if a dataset reports `n_failed > 0`, a
completion marker is missing, vLLM returns HTTP 400/500, Track A rendering is
blank, or Track C starts a Slicer-related process.

## Step 8: Run Full Experiments

Track A MedGemma Table 1 rows:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --model-set medgemma \
  --limit 0
```

Track A all configured rows:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --model-set all \
  --limit 0
```

Track C MedGemma compressed-figure rows:

```bash
bash radiology/track_c/scripts/reproduce_track_c.sh \
  --model-set medgemma \
  --method compressed_figures \
  --limit 0
```

Track C all configured compressed-figure rows:

```bash
bash radiology/track_c/scripts/reproduce_track_c.sh \
  --model-set all \
  --method compressed_figures \
  --limit 0
```

To run the alternate sampled-slice method, use `--method table3_api`. Keep its
results separate; the launcher writes under the method-specific directory.

`--limit 0` means:

```text
LUMIERE: 139 questions
UCSF-PDGM: 495 cases
NSCLC: 162 cases
```

Both launchers resume matching completed model/dataset pairs by default. Use
`--no-resume` only for an intentional clean rerun.

## Step 9: Validate Results

Track A requires root-level `table1_results.csv/json` and `.complete` markers
under every selected model's `lumiere`, `ucsf_pdgm`, and `nsclc` directories.

Track C requires `track_c_results.csv/json` under
`TRACK_C_RESULTS_ROOT/<method>/` and the same three model/dataset completion
markers.

For each latest dataset run, require:

```text
summary.json
Paper_Table_Data.json
model output and prompt provenance
trace/log files
evidence artifacts
```

Programmatically confirm:

- `n_failed == 0`
- count equals the requested limit or 139/495/162 for full runs
- requested model, dataset, track, and method match
- aggregate and per-case result files are nonempty
- every artifact path declared in a per-case result exists; zero-action
  `trace.jsonl` and `evidence.jsonl` files may be empty but must be present
- static prompts came from `shared/prompts/`
- Track C started no Slicer, WebServer, or Xvfb process
- cleanup released ports and reproduction-owned processes

For NSCLC Track C, report
`segment_metrics.n_cases_with_segment_gt` with strict/localization scores. A
zero denominator is a dataset-evidence limitation, not permission to invent a
score.

Scan the selected result root:

```bash
grep -RIE \
  'Traceback \(most recent call last\)|HTTP/1\.1 400|HTTP/1\.1 500|exceeds the max_model_len|Tool failed:' \
  /absolute/path/to/results --include='*.log' --include='*.jsonl' || true
```

Low accuracy or missing answer fields are model behavior, not automatically
an infrastructure failure. Report them without rewriting outputs or scores.

## Failure Routing

- **Wrong Python or missing ensurepip:** select Python 3.10 with venv support.
- **Package conflict:** recreate the environment from the audited lock and run
  `pip check`.
- **vLLM OOM:** stop stale GPU processes, then record any tensor parallel,
  memory utilization, quantization, or model changes.
- **Too many open files:** if this error actually occurs, raise `ulimit -n`
  within site policy and record the value; a startup warning alone is not a
  failed smoke test.
- **Context overflow:** retain the tested image/history/result limits.
- **Home quota failure:** move all caches, runtime, results, and weights to
  project or scratch storage.
- **Track A cannot start:** inspect Slicer/Xvfb logs, GLX, `DISPLAY`, libraries,
  executable bits, and extension verification.
- **Track A blank evidence:** fix the Qt/OpenGL layout; do not use
  `--no-main-window`.
- **Track C imports Slicer:** stop and fix the code/configuration boundary.
- **Dataset count mismatch:** verify release, extraction, metadata, and filters;
  never alter expected counts.
- **M3D/RadFM failure:** use the dedicated upstream environment and direct
  worker commands in `shared/docs/track_c.md`.
- **Failed case:** inspect `summary.json`, runlog, model-server log, prompt
  provenance, and evidence before changing code.

## Final Report

Report the source revision/checksum; selected scope; OS/GPU/Python/vLLM/Slicer
versions; models and non-secret overrides; dataset roots and counts; exact
commands; Track A and Track C result paths; per-dataset `n_failed`; aggregate
files; warnings versus errors; process cleanup; and every deviation from the
audited protocol.

Only call a requested track complete when all of its validation criteria pass.
Only call the radiology release ready when both Track A and Track C limit-5
smoke workflows pass.
