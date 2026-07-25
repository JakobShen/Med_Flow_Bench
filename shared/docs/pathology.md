# Pathology Runtime (QuPath)

This document describes the viewer-native pathology implementation used for
BRACS and CAMELYON17. It is organized independently from the radiology track
structure.

## Architecture

The host model is the agent. Python owns the bounded tool loop, prompt,
observations, run state, evidence, and structured logs:

```text
benchmark runner
  -> qupath_agent_engine host loop
  -> OpenAI or Gemini function calling
  -> deterministic QuPath tools
  -> QuPath Java attach bridge
  -> viewer screenshots and structured state
  -> deterministic pathology scorer
```

The model receives the current QuPath viewer screenshot and a structured state
summary after each step. Direct controls are preferred for loading, image type,
zoom, pan, state query, capture, and coordinate conversion.

Midscene is an optional GUI-only sidecar. It does not plan the benchmark and
must not interpret pathology content. The benchmark runners disable its tools
by default.

## Requirements

- Python 3.10.x; the audited environment used Python 3.10.20
- QuPath 0.7.x
- Java 17+ JDK with `java`, `javac`, and `jar`
- macOS Accessibility and screen-recording permission for the terminal process
- OpenAI or Gemini credentials
- Node.js and npm only when enabling the optional Midscene sidecar

Install:

```bash
bash shared/setup/create_python_env.sh pathology
source .venv-pathology/bin/activate

export QUPATH_EXECUTABLE=/absolute/path/to/QuPath
```

The current screen-capture and Java attach flow is verified on macOS. Windows
launch support is partial. Linux desktop control is not implemented.

## Direct-Control Bridge

`engines/qupath_agent_engine/software/qupath/java/` contains Java source, not
compiled artifacts. On first use, the launcher compiles an attach helper under the
user's cache directory and attaches it to the local QuPath JVM. This permits
deterministic control on the JavaFX thread without clicking through menus.

Only run trusted copies of this code. Java attach has the same local authority
as the QuPath process.

The complete reusable tool registry is in
`engines/qupath_agent_engine/agent/tooling.py`. Benchmark runners expose a
restricted subset: zoom, pan, focus, capture, screenshot-to-WSI coordinate
conversion, and sleep.

## Host Models

OpenAI:

```bash
export WORKSTATION_AGENT_PROVIDER=openai
export WORKSTATION_AGENT_MODEL=gpt-5.4-mini
export OPENAI_API_KEY=...
```

Gemini Developer API:

```bash
export WORKSTATION_AGENT_PROVIDER=gemini
export WORKSTATION_AGENT_MODEL=gemini-2.5-pro
export GEMINI_API_KEY=...
```

Gemini on Vertex AI:

```bash
export WORKSTATION_AGENT_PROVIDER=gemini
export WORKSTATION_AGENT_MODEL=gemini-3-flash-preview
export WORKSTATION_AGENT_GEMINI_BACKEND=vertexai
export VERTEX_API_KEY=...
```

Application-default credentials are also supported through
`GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION`. Keep all credentials in
the environment or an untracked `.env`; never add them to the repository.

## Optional Midscene Sidecar

```bash
cd pathology/runtime/qupath_sidecar
npm ci
npx --no-install tsx src/cli.ts env-check
cd ../../..

export MIDSCENE_QUPATH_SIDECAR_DIR="$PWD/pathology/runtime/qupath_sidecar"
export MIDSCENE_MODEL_BASE_URL=https://provider.example/v1
export MIDSCENE_MODEL_API_KEY=...
export MIDSCENE_MODEL_NAME=...
export MIDSCENE_MODEL_FAMILY=...
```

Pass `--no-disable-gui-agent` to a runner only when the sidecar is configured.
This expands the desktop-control surface and may transmit GUI screenshots to a
separate model provider.

## Data Layout

BRACS:

```text
/path/to/BRACS/
  BRACS.xlsx
  BRACS_WSI/
    ... .svs ...
  BRACS_WSI_Annotations/
    ... .qpdata ...
```

The loader discovers `.svs` and `.qpdata` recursively using exact case
filenames. The default `--forced_qpdata` setting excludes cases without
annotation projects so evidence remains scoreable.

CAMELYON17:

```text
/path/to/CAMELYON17/
  stages.csv
  ... nested .tif slide images ...
  ... nested masks and/or XML annotations ...
```

The loader discovers slides and evidence recursively. The default benchmark
includes only evidence-backed slides. This implementation evaluates
slide-level tumor presence and largest-metastasis category, not patient-level
pN stage.

Download helpers:

```bash
DATA_ROOT=/path/to/data \
  bash shared/data_download/download_pathology_collections.sh smoke

DATA_ROOT=/path/to/data \
  bash shared/data_download/download_pathology_collections.sh resume
```

Dataset-host terms remain controlling. Do not redistribute the downloaded
content based only on the code license.

## Benchmarks

BRACS:

```bash
python pathology/benchmarks/benchmark_bracs.py \
  --data-root /path/to/BRACS \
  --qupath-executable "$QUPATH_EXECUTABLE" \
  --provider openai \
  --model gpt-5.4-mini \
  --max-steps 20 \
  --set Testing
```

Useful BRACS selectors are `--case-id`, repeated `--set`, `--limit`,
`--balance-subtypes`, and `--forced_qpdata`.

CAMELYON17:

```bash
python pathology/benchmarks/benchmark_camelyon17.py \
  --data-root /path/to/CAMELYON17 \
  --qupath-executable "$QUPATH_EXECUTABLE" \
  --provider openai \
  --model gpt-5.4-mini \
  --max-steps 20
```

Useful CAMELYON17 selectors are `--case-id`, `--patient-id`, `--center`,
`--slide-label`, `--limit`, `--balance-subtypes`, and
`--include-no-evidence`.

Start with `--limit 1`. Use `--restart-between-cases` when isolation is more
important than runtime, and `--leave-open` only for supervised debugging.

## Demo And Smoke Check

The non-LLM check exercises launch, state query, image type, zoom, pan, and
capture:

```bash
python pathology/examples/qupath_smoke_test.py \
  --image /path/to/slide.svs \
  --qupath-executable "$QUPATH_EXECUTABLE" \
  --save
```

The interactive agent accepts an image or project:

```bash
python pathology/examples/demo_qupath.py \
  --image /path/to/slide.svs \
  --task "Inspect the slide and identify the most relevant tissue region" \
  --provider openai \
  --model gpt-5.4-mini \
  --max-steps 20 \
  --save
```

## Outputs And Scoring

Each benchmark creates a timestamped root below `--runs-dir`. Per-case outputs
include:

```text
logs/runlog.jsonl
evidence/*.png
summary.json
benchmark_result.json
judge_log.json
```

The run log is the canonical model/tool trace. Benchmark roots also contain a
manifest, aggregate output, and paper-oriented summary data.

BRACS requires one of `N`, `PB`, `UDH`, `FEA`, `ADH`, `DCIS`, or `IC`.
Non-`N` answers require one or more full-resolution WSI coordinates and are
scored against hidden annotation regions.

CAMELYON17 requires tumor presence and one of `negative`, `itc`, `micro`, or
`macro`. Positive answers require localizations scored against masks or XML
annotations. Hidden references are never included in the model-facing prompt
or viewer state.

## Operational Safety

- Run QuPath in a dedicated desktop session.
- Keep Java attach and Midscene local; expose neither as a network service.
- Do not enable GUI tools unless the sidecar and its provider are configured.
- Treat run directories as sensitive because they can contain screenshots,
  model text, metadata, and local paths.
- Close unrelated applications before enabling desktop GUI control.
