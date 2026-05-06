# Medical Benchmark Public Subset

This repository contains the public benchmark subset used for dataset loading,
deterministic judging, report-table aggregation, and Track3-style model
evaluation.

## Contents

- Dataset loaders for UCSF-PDGM, NSCLC Radiogenomics, and LUMIERE.
- Rule-based answer extraction and scoring utilities.
- Hidden-evidence scoring helpers for mask-derived key slices and RAS points.
- Track3 benchmark runners for UCSF-PDGM, NSCLC, and LUMIERE.
- API-backed Track3 inference helpers plus M3D and RadFM worker wrappers.
- Sanitized public dataset download helpers for the benchmark datasets.

## Layout

```text
scripts/
  benchmark_UCSF_PDGM_track3.py
  benchmark_NSCLC_track3.py
  benchmark_LUMIERE_track3.py
  download_data/
  track3_workers/
slicer_agent_engine/
  benchmarking/
  track3/
  llm/
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-track3.txt
pip install -e .
```

For API-backed baselines, set provider credentials in your shell or secret
manager before running a benchmark:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export LLM_PROVIDER="openai"
export LLM_MODEL="gpt-5-mini"
```

## Example Track3 Runs

UCSF-PDGM:

```bash
python scripts/benchmark_UCSF_PDGM_track3.py \
  --data_root /path/to/UCSF-PDGM \
  --csv_path /path/to/UCSF-PDGM-metadata_v5.csv \
  --method compressed_figures \
  --provider openai \
  --model gpt-5-mini \
  --limit 20
```

NSCLC:

```bash
python scripts/benchmark_NSCLC_track3.py \
  --data_root /path/to/nsclc-radiogenomics \
  --metadata_csv metadata.csv \
  --labels_csv /path/to/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv \
  --method compressed_figures \
  --provider openai \
  --model gpt-5-mini \
  --limit 20
```

LUMIERE:

```bash
python scripts/benchmark_LUMIERE_track3.py \
  --data_root /path/to/LUMIERE \
  --ratings_csv LUMIERE-ExpertRating-v202211.csv \
  --datacompleteness_csv LUMIERE-datacompleteness.csv \
  --method compressed_figures \
  --provider openai \
  --model gpt-5-mini \
  --limit_question 20
```

## Dataset Download Helpers

Download helper scripts are provided under `scripts/download_data/`. They use
placeholder paths by default and can be redirected with environment variables:

```bash
DATA_ROOT=/path/to/data \
  bash scripts/download_data/download_tcia_radiology_collections.sh smoke

DATA_ROOT=/path/to/data \
  bash scripts/download_data/download_pathology_collections.sh smoke

OUT_DIR=/path/to/LUMIERE_unregistered \
  bash scripts/download_data/download_lumiere_unregistered.sh
```

Use `resume` instead of `smoke` for the collection scripts to start resumable
downloads.

## Outputs

Each run writes a timestamped output directory containing:

- `benchmark_track3.log`
- `results.jsonl`
- `summary.json`
- `Paper_Table_Data.json`
- per-case `agent_output.txt`, `result.json`, trace, and generated artifacts

## Notes

Paths in the examples are placeholders. Keep local datasets, model weights,
run logs, and credentials outside version control.
