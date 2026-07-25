# Runtime-Independent Track C

Track C is a radiology benchmark path, not a 3D Slicer component. It contains
dataset loaders, deterministic scoring, artifact construction, API-backed
multimodal inference, and external worker wrappers for open medical VLM
baselines. It does not start or communicate with Slicer or WebServer.

## Entry Points

- `radiology/track_c/benchmark_UCSF_PDGM_track_c.py`
- `radiology/track_c/benchmark_NSCLC_track_c.py`
- `radiology/track_c/benchmark_LUMIERE_track_c.py`

## Shared Modules

- `slicer_agent_engine.benchmarking.*`: dataset loaders, answer extraction, mask evidence scoring, and paper-table export.
- `radiology_track_c.volumes`: NIfTI and DICOM volume loading helpers.
- `radiology_track_c.artifacts`: compressed figures and sampled-slice artifacts.
- `radiology_track_c.api_inference`: API-backed multimodal model calls.
- `radiology_track_c.external`: subprocess wrappers for external workers.
- `radiology_track_c.segment_gt`: mask-derived ground-truth preparation for scoring utilities.

The reused `slicer_agent_engine.benchmarking`, `llm`, and `session` modules are
host-side Python utilities. Track C does not import `slicer_client`,
`slicer_launcher`, `app.bootstrap`, or any bridge module.

## Methods

`compressed_figures` creates montage-style figures from selected planes. This
is the paper's 2D slice-montage Track C baseline.

`table3_api` samples a fixed number of slices per selected plane.

`m3d` launches `radiology/track_c/workers/m3d_worker.py` in a separate Python environment.

`radfm` launches `radiology/track_c/workers/radfm_worker.py` with a RadFM checkout and checkpoint.

## One-Command API Reproduction

Create the Track C environment and untracked configuration:

```bash
bash shared/setup/create_python_env.sh track_c
bash radiology/model_serving/create_vllm_env.sh .venv-vllm
cp shared/config/track_c.env.example .env.track_c
```

Fill the dataset paths, `PYTHON_BIN`, `VLLM_BIN`, and either a local model path
or `HF_TOKEN`. Track C does not use `SLICER_EXECUTABLE`, `DISPLAY`, Xvfb,
WebServer, or Slicer extensions.

The default command starts MedGemma 1.5 4B with vLLM, runs five units from each
dataset using `compressed_figures`, validates every output, stops vLLM, and
aggregates the paper metrics:

```bash
bash radiology/track_c/scripts/reproduce_track_c.sh
```

Use the full eligible protocol with:

```bash
bash radiology/track_c/scripts/reproduce_track_c.sh --limit 0
```

Select another released vLLM model through the shared manifest:

```bash
bash radiology/track_c/scripts/reproduce_track_c.sh \
  --model qwen3.5-9b \
  --limit 5
```

The launcher supports `compressed_figures` and `table3_api`. M3D and RadFM
have incompatible upstream environments and are run directly with the worker
arguments below instead of through the vLLM launcher.

## Example

```bash
python radiology/track_c/benchmark_UCSF_PDGM_track_c.py \
  --data_root /path/to/UCSF-PDGM \
  --csv_path /path/to/UCSF-PDGM-metadata_v5.csv \
  --method compressed_figures \
  --provider openai \
  --model gpt-5-mini \
  --limit 20
```

## Outputs

Run roots contain `summary.json`, `results.jsonl`, `Paper_Table_Data.json`, and
`benchmark_track_c.log`. Per-case folders contain model text output, result JSON,
trace files, and generated image artifacts. The one-command launcher writes
aggregate `track_c_results.csv` and `track_c_results.json` files under
`TRACK_C_RESULTS_ROOT/<method>/`.

## NSCLC Evidence Ground Truth

NSCLC task metrics use the public clinical labels. Strict and localization
metrics use the collection's public primary-tumor DICOM SEG objects directly;
Track C decodes their frame geometry with `pydicom` and does not require Slicer
or a separate conversion step. The collection has known generic or incorrect
segment labels such as `Heart` and `Tissue`, so the decoder does not use label
text to decide whether a public SEG is a tumor mask. It records the original
label in `segment_gt.json` for auditing.

An aligned case-level NIfTI tumor mask placed directly inside a case directory
takes precedence over DICOM SEG. Accepted `.nii` or `.nii.gz` names contain
`segmentation`, `tumor_mask`, `mask`, `label`, or `seg`.

Not every public case has a usable segmentation. Always report
`segment_metrics.n_cases_with_segment_gt` with strict/localization metrics. If
the denominator is zero, the run skips evidence scoring, emits zero
placeholders, and the one-command launcher prints a warning.

Dataset and annotation details are provided by
[TCIA NSCLC-Radiogenomics](https://www.cancerimagingarchive.net/collection/nsclc-radiogenomics/).
The label issue is documented in the
[Imaging Data Commons discussion](https://discourse.canceridc.dev/t/mislabeled-segmentations-for-nsclc-radiogenomics/518).
