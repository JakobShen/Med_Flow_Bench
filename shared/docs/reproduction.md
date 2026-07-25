# Reproduction Commands

All paths below are placeholders. Install the package and run these commands
from the repository root after exporting provider credentials.

For the full viewer-native radiology Table 1, use the model manifest and
resume-aware launcher instead of invoking each row manually:

```bash
cp shared/config/table1.env.example .env.table1
# Fill .env.table1 once, then:
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh --limit 0
```

Omit `--limit 0` for the default five-unit-per-dataset end-to-end check.
Use `--model-set open` for the Qwen, Gemma, MedGemma, and Lingshu vLLM rows.
The vLLM setup and direct Track A/Track C commands are documented in
[`open_weight_models.md`](open_weight_models.md).

```bash
export PROVIDER=openai
export MODEL=gpt-5-mini

export UCSF_ROOT=/path/to/data/ucsf-pdgm
export UCSF_CSV=/path/to/data/ucsf-pdgm/UCSF-PDGM-metadata_v5.csv

export NSCLC_ROOT=/path/to/data/nsclc-radiogenomics
export NSCLC_METADATA=/path/to/data/nsclc-radiogenomics/metadata.csv
export NSCLC_LABELS=/path/to/data/nsclc-radiogenomics/NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv

export LUMIERE_ROOT=/path/to/data/LUMIERE
export LUMIERE_NATIVE_ROOT=/path/to/data/LUMIERE_unregistered
export LUMIERE_RATINGS=LUMIERE-ExpertRating-v202211.csv
export LUMIERE_COMPLETENESS=LUMIERE-datacompleteness.csv

export BRACS_ROOT=/path/to/data/BRACS
export CAMELYON17_ROOT=/path/to/data/CAMELYON17
export QUPATH_EXECUTABLE=/absolute/path/to/QuPath
```

## Track A: Viewer-Native

Track A uses the shared Slicer runners without optional toolpack flags.

UCSF-PDGM:

```bash
python radiology/slicer_track_a_b/benchmark_UCSF_PDGM.py \
  --data_root "$UCSF_ROOT" \
  --csv_path "$UCSF_CSV" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20 \
  --limit 100
```

NSCLC Radiogenomics:

```bash
python radiology/slicer_track_a_b/benchmark_NSCLC.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20 \
  --limit 100
```

LUMIERE registered image space:

```bash
python radiology/slicer_track_a_b/benchmark_LUMIERE.py \
  --data_root "$LUMIERE_ROOT" \
  --ratings_csv "$LUMIERE_RATINGS" \
  --datacompleteness_csv "$LUMIERE_COMPLETENESS" \
  --lumiere_image_space registered \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20
```

## Pathology With QuPath

The pathology experiments use QuPath and the separate `qupath_agent_engine`
host loop. They do not use radiology Track A/B/C labels, the Slicer runner, or
radiology toolpacks.

Set the QuPath host model independently:

```bash
export WORKSTATION_AGENT_PROVIDER=openai
export WORKSTATION_AGENT_MODEL=gpt-5.4-mini
export OPENAI_API_KEY=...
```

BRACS Testing split:

```bash
python pathology/benchmarks/benchmark_bracs.py \
  --data-root "$BRACS_ROOT" \
  --qupath-executable "$QUPATH_EXECUTABLE" \
  --provider "$WORKSTATION_AGENT_PROVIDER" \
  --model "$WORKSTATION_AGENT_MODEL" \
  --max-steps 20 \
  --set Testing
```

One BRACS case:

```bash
python pathology/benchmarks/benchmark_bracs.py \
  --data-root "$BRACS_ROOT" \
  --qupath-executable "$QUPATH_EXECUTABLE" \
  --provider "$WORKSTATION_AGENT_PROVIDER" \
  --model "$WORKSTATION_AGENT_MODEL" \
  --max-steps 20 \
  --case-id BRACS_CASE_ID \
  --limit 1
```

CAMELYON17 slide-level evaluation:

```bash
python pathology/benchmarks/benchmark_camelyon17.py \
  --data-root "$CAMELYON17_ROOT" \
  --qupath-executable "$QUPATH_EXECUTABLE" \
  --provider "$WORKSTATION_AGENT_PROVIDER" \
  --model "$WORKSTATION_AGENT_MODEL" \
  --max-steps 20
```

One CAMELYON17 slide:

```bash
python pathology/benchmarks/benchmark_camelyon17.py \
  --data-root "$CAMELYON17_ROOT" \
  --qupath-executable "$QUPATH_EXECUTABLE" \
  --provider "$WORKSTATION_AGENT_PROVIDER" \
  --model "$WORKSTATION_AGENT_MODEL" \
  --max-steps 20 \
  --case-id patient_000_node_4 \
  --limit 1
```

Both runners disable Midscene GUI tools by default and use deterministic
QuPath controls. See `shared/docs/pathology.md` for optional GUI configuration,
selectors, output files, and scoring semantics.

## Track B: Advanced Operations

Track B uses the same three Slicer runners as Track A. The experiment becomes
Track B only when an optional operation pack is provided:

| Advanced condition | Supported benchmark | Required flags | External prerequisite |
| --- | --- | --- | --- |
| Curated segmentation | UCSF-PDGM, NSCLC | `--segmentation_toolpacks` | Slicer Segment Editor and SegmentEditorExtraEffects |
| Registration | LUMIERE unregistered | `--registration_toolpacks --lumiere_image_space unregistered` | BRAINSFit/resampling modules; SlicerElastix only for the optional Elastix path |
| MONAI Label | UCSF-PDGM, NSCLC | `--monai_label_toolpacks` | Reachable MONAI Label-compatible service |
| VISTA3D preset | NSCLC | `--vista` | VISTA3D service configured as documented in `environments.md` |
| All optional packs | Any Slicer runner | `--all_toolpacks` | Every selected pack's prerequisite |

Use `--activate_toolpacks manual` to test whether the agent discovers and
activates a provided pack. Use `--activate_toolpacks auto` to expose its tools
at the first model turn. Do not add any of these flags for Track A.

UCSF-PDGM with curated segmentation tools:

```bash
python radiology/slicer_track_a_b/benchmark_UCSF_PDGM.py \
  --data_root "$UCSF_ROOT" \
  --csv_path "$UCSF_CSV" \
  --segmentation_toolpacks \
  --activate_toolpacks auto \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20 \
  --limit 100
```

NSCLC with curated segmentation tools:

```bash
python radiology/slicer_track_a_b/benchmark_NSCLC.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --segmentation_toolpacks \
  --activate_toolpacks auto \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20 \
  --limit 100
```

UCSF-PDGM with MONAI Label/VISTA3D:

```bash
python radiology/slicer_track_a_b/benchmark_UCSF_PDGM.py \
  --data_root "$UCSF_ROOT" \
  --csv_path "$UCSF_CSV" \
  --monai_label_toolpacks \
  --activate_toolpacks auto \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20 \
  --limit 100
```

NSCLC with the VISTA3D benchmark preset:

```bash
python radiology/slicer_track_a_b/benchmark_NSCLC.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --vista \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20 \
  --limit 100
```

LUMIERE native image space with registration tools:

```bash
python radiology/slicer_track_a_b/benchmark_LUMIERE.py \
  --data_root "$LUMIERE_NATIVE_ROOT" \
  --ratings_csv "$LUMIERE_RATINGS" \
  --datacompleteness_csv "$LUMIERE_COMPLETENESS" \
  --lumiere_image_space unregistered \
  --registration_toolpacks \
  --activate_toolpacks auto \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20
```

The registration workflow is expected to select the intended OLD and CURRENT
volumes, run BRAINSFit, resample/apply the transform, and verify the registered
fusion rather than the original moving volume.

## Track C: Runtime-Agnostic

Track C does not start or connect to 3D Slicer.

For the API-backed `compressed_figures` paper baseline, configure dataset and
model paths once and run all three datasets:

```bash
cp shared/config/track_c.env.example .env.track_c
bash radiology/track_c/scripts/reproduce_track_c.sh
```

This defaults to MedGemma 1.5 4B and five units per dataset. Use `--limit 0`
for the full 139/495/162 eligible-unit protocol, `--model KEY` for another
manifest model, or `--method table3_api` for sampled slices. See
[`track_c.md`](track_c.md) for environment and output details.

### UCSF-PDGM

Compressed figures:

```bash
python radiology/track_c/benchmark_UCSF_PDGM_track_c.py \
  --data_root "$UCSF_ROOT" \
  --csv_path "$UCSF_CSV" \
  --method compressed_figures \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --limit 100
```

Table-3 sampled slices:

```bash
python radiology/track_c/benchmark_UCSF_PDGM_track_c.py \
  --data_root "$UCSF_ROOT" \
  --csv_path "$UCSF_CSV" \
  --method table3_api \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --table3_planes axial \
  --table3_sample_slices 10 \
  --limit 100
```

M3D:

```bash
python radiology/track_c/benchmark_UCSF_PDGM_track_c.py \
  --data_root "$UCSF_ROOT" \
  --csv_path "$UCSF_CSV" \
  --method m3d \
  --m3d_python "$M3D_PYTHON" \
  --m3d_model_name_or_path "$M3D_MODEL_PATH" \
  --m3d_device cuda \
  --limit 100
```

RadFM:

```bash
python radiology/track_c/benchmark_UCSF_PDGM_track_c.py \
  --data_root "$UCSF_ROOT" \
  --csv_path "$UCSF_CSV" \
  --method radfm \
  --radfm_python "$RADFM_PYTHON" \
  --radfm_repo_root "$RADFM_REPO_ROOT" \
  --radfm_checkpoint "$RADFM_CHECKPOINT" \
  --radfm_device cuda \
  --limit 100
```

### NSCLC Radiogenomics

Compressed figures:

```bash
python radiology/track_c/benchmark_NSCLC_track_c.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --method compressed_figures \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --limit 100
```

Table-3 sampled slices:

```bash
python radiology/track_c/benchmark_NSCLC_track_c.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --method table3_api \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --table3_planes axial \
  --table3_sample_slices 10 \
  --limit 100
```

M3D:

```bash
python radiology/track_c/benchmark_NSCLC_track_c.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --method m3d \
  --m3d_python "$M3D_PYTHON" \
  --m3d_model_name_or_path "$M3D_MODEL_PATH" \
  --m3d_device cuda \
  --m3d_sequence_strategy all_vote \
  --limit 100
```

RadFM:

```bash
python radiology/track_c/benchmark_NSCLC_track_c.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --method radfm \
  --radfm_python "$RADFM_PYTHON" \
  --radfm_repo_root "$RADFM_REPO_ROOT" \
  --radfm_checkpoint "$RADFM_CHECKPOINT" \
  --radfm_device cuda \
  --radfm_sequence_strategy all_vote \
  --limit 100
```

### LUMIERE

Compressed figures:

```bash
python radiology/track_c/benchmark_LUMIERE_track_c.py \
  --data_root "$LUMIERE_ROOT" \
  --ratings_csv "$LUMIERE_RATINGS" \
  --datacompleteness_csv "$LUMIERE_COMPLETENESS" \
  --method compressed_figures \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --limit_question 100
```

Table-3 sampled slices:

```bash
python radiology/track_c/benchmark_LUMIERE_track_c.py \
  --data_root "$LUMIERE_ROOT" \
  --ratings_csv "$LUMIERE_RATINGS" \
  --datacompleteness_csv "$LUMIERE_COMPLETENESS" \
  --method table3_api \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --table3_planes axial \
  --table3_sample_slices 10 \
  --limit_question 100
```

M3D:

```bash
python radiology/track_c/benchmark_LUMIERE_track_c.py \
  --data_root "$LUMIERE_ROOT" \
  --ratings_csv "$LUMIERE_RATINGS" \
  --datacompleteness_csv "$LUMIERE_COMPLETENESS" \
  --method m3d \
  --m3d_python "$M3D_PYTHON" \
  --m3d_model_name_or_path "$M3D_MODEL_PATH" \
  --m3d_device cuda \
  --limit_question 100
```

RadFM:

```bash
python radiology/track_c/benchmark_LUMIERE_track_c.py \
  --data_root "$LUMIERE_ROOT" \
  --ratings_csv "$LUMIERE_RATINGS" \
  --datacompleteness_csv "$LUMIERE_COMPLETENESS" \
  --method radfm \
  --radfm_python "$RADFM_PYTHON" \
  --radfm_repo_root "$RADFM_REPO_ROOT" \
  --radfm_checkpoint "$RADFM_CHECKPOINT" \
  --radfm_device cuda \
  --limit_question 100
```

## Focused Case Runs

Use existing selector arguments rather than editing source:

```bash
python radiology/slicer_track_a_b/benchmark_LUMIERE.py \
  --data_root "$LUMIERE_NATIVE_ROOT" \
  --ratings_csv "$LUMIERE_RATINGS" \
  --datacompleteness_csv "$LUMIERE_COMPLETENESS" \
  --lumiere_image_space unregistered \
  --registration_toolpacks \
  --activate_toolpacks auto \
  --question_id Patient-022__week-045__to__week-053 \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20

python radiology/slicer_track_a_b/benchmark_NSCLC.py \
  --data_root "$NSCLC_ROOT" \
  --metadata_csv "$NSCLC_METADATA" \
  --labels_csv "$NSCLC_LABELS" \
  --case_ids R01-001 \
  --provider "$PROVIDER" \
  --model "$MODEL" \
  --max_rounds 20
```
