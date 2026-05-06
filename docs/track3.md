# Track3 Benchmark Runners

The public Track3 code path contains dataset loaders, deterministic scoring,
artifact construction, API-backed multimodal inference, and external worker
wrappers for open medical VLM baselines.

## Entry Points

- `scripts/benchmark_UCSF_PDGM_track3.py`
- `scripts/benchmark_NSCLC_track3.py`
- `scripts/benchmark_LUMIERE_track3.py`

## Shared Modules

- `slicer_agent_engine.benchmarking.*`: dataset loaders, answer extraction, mask evidence scoring, and paper-table export.
- `slicer_agent_engine.track3.volumes`: NIfTI and DICOM volume loading helpers.
- `slicer_agent_engine.track3.artifacts`: compressed figures and sampled-slice artifacts.
- `slicer_agent_engine.track3.api_inference`: API-backed multimodal model calls.
- `slicer_agent_engine.track3.external`: subprocess wrappers for external workers.
- `slicer_agent_engine.track3.segment_gt`: mask-derived ground-truth preparation for public scoring utilities.

## Methods

`compressed_figures` creates montage-style figures from selected planes.

`table3_api` samples a fixed number of slices per selected plane.

`m3d` launches `scripts/track3_workers/m3d_worker.py` in a separate Python environment.

`radfm` launches `scripts/track3_workers/radfm_worker.py` with a RadFM checkout and checkpoint.

## Example

```bash
python scripts/benchmark_UCSF_PDGM_track3.py \
  --data_root /path/to/UCSF-PDGM \
  --csv_path /path/to/UCSF-PDGM-metadata_v5.csv \
  --method compressed_figures \
  --provider openai \
  --model gpt-5-mini \
  --limit 20
```

## Outputs

Run roots contain `summary.json`, `results.jsonl`, `Paper_Table_Data.json`, and
`benchmark_track3.log`. Per-case folders contain model text output, result JSON,
trace files, and generated image artifacts.
