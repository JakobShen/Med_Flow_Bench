# Dataset Download Helpers

This directory contains sanitized helpers for public benchmark datasets. The
scripts never contain credentials or fixed local paths. Dataset licenses and
upstream terms still apply.

Run these commands from the repository root. Use an absolute path for every
output directory.

## Radiology: What Each Mode Does

`download_tcia_radiology_collections.sh` covers the public
NSCLC-Radiogenomics, UCSF-PDGM, and LUMIERE release endpoints:

| Mode | Network activity | Result |
| --- | --- | --- |
| `smoke` | Small HTTP requests only | Checks endpoints and writes temporary URL manifests. It does **not** download imaging data. |
| `resume` | Full dataset downloads | Downloads or resumes the three public collections. Expect large transfers and substantial disk use. |
| `organize` | None | Re-runs the lightweight UCSF metadata-linking step. |

Required commands are `python3`, one of `aria2c`/`wget`/`curl`, and Java for
the NSCLC NBIA retriever. The script also needs `dpkg-deb` or `ar` to unpack a
private copy of the NBIA client. It does not install system packages.

Check all public endpoints without downloading imaging:

```bash
DATA_ROOT=/absolute/path/to/data \
  bash shared/data_download/download_tcia_radiology_collections.sh smoke
```

Start or resume the complete download:

```bash
DATA_ROOT=/absolute/path/to/data \
  WORK_ROOT=/absolute/path/to/download-work \
  bash shared/data_download/download_tcia_radiology_collections.sh resume
```

Partial files are retained intentionally so the same command can resume them.
The outputs are:

```text
/absolute/path/to/data/
  nsclc-radiogenomics/  # NBIA DICOM collection
  ucsf-pdgm/.flat/      # downloaded UCSF NIfTI release assets
  LUMIERE/              # downloaded Figshare release files and archives
```

The benchmark also requires public clinical metadata/labels. Supply the
official `UCSF-PDGM-metadata_v5.csv` with `UCSF_METADATA_CSV` (or pass its path
directly to the runner). For NSCLC, pass the downloaded public collection
metadata and label CSV paths to `--metadata_csv` and `--labels_csv`. These
clinical files are separate from the NBIA image manifest.

The combined helper downloads UCSF assets into `.flat` and leaves LUMIERE ZIP
files unextracted. Preserve the upstream data, then prepare the directory
layout shown in the root README before launching a benchmark.

## LUMIERE Registered And Unregistered Data

The dedicated LUMIERE helper downloads the public Figshare release, verifies
the published MD5 values, and can extract a benchmark-ready image tree. The
upstream imaging ZIP is approximately 30 GB, and extraction needs additional
free space.

For native/unregistered images used by the registration experiment:

```bash
OUT_DIR=/absolute/path/to/data/LUMIERE_unregistered \
  EXTRACT_MODE=native_only \
  bash shared/data_download/download_lumiere_unregistered.sh
```

For the complete release, including registered atlas-space files:

```bash
OUT_DIR=/absolute/path/to/data/LUMIERE \
  EXTRACT_MODE=full \
  bash shared/data_download/download_lumiere_unregistered.sh
```

Use `EXTRACT_MODE=none` to download and verify without extraction. Downloads
are stored below `$OUT_DIR/.downloads`; rerunning the same command resumes an
interrupted transfer. Set `KEEP_ZIP=0` only when the verified archive is no
longer needed.

## Pathology Collections

The pathology helper covers CAMELYON17 and BRACS:

```bash
DATA_ROOT=/path/to/data \
  bash shared/data_download/download_pathology_collections.sh smoke

DATA_ROOT=/path/to/data \
  bash shared/data_download/download_pathology_collections.sh resume
```

It requires `curl`, `wget`, `unzip`, and the AWS CLI. On Linux x86_64, the
script can install a private AWS CLI copy under `.download_work/` when no
system installation exists. On other platforms, install the AWS CLI first.

These scripts only automate access to public endpoints. Dataset terms,
attribution requirements, and redistribution restrictions still apply.
