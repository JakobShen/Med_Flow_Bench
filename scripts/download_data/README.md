# Dataset Download Helpers

This directory contains sanitized helper scripts for downloading the public
datasets used by the benchmark subset. All paths are configurable through
environment variables and default to local directories under the current
working directory.

## Radiology Collections

```bash
DATA_ROOT=/path/to/data \
  bash scripts/download_data/download_tcia_radiology_collections.sh smoke

DATA_ROOT=/path/to/data \
  bash scripts/download_data/download_tcia_radiology_collections.sh resume
```

The radiology helper covers NSCLC-Radiogenomics, UCSF-PDGM, and LUMIERE public
release files. Use `organize` to run lightweight post-download organization
steps without starting new downloads.

## Pathology Collections

```bash
DATA_ROOT=/path/to/data \
  bash scripts/download_data/download_pathology_collections.sh smoke

DATA_ROOT=/path/to/data \
  bash scripts/download_data/download_pathology_collections.sh resume
```

The pathology helper covers CAMELYON17 and BRACS public release locations.

## LUMIERE Native Image-Space Files

```bash
OUT_DIR=/path/to/LUMIERE_unregistered \
  bash scripts/download_data/download_lumiere_unregistered.sh
```

Set `EXTRACT_MODE=full` to extract the complete imaging archive, or
`EXTRACT_MODE=none` to download and verify files without extraction.
