# MedFlowBench Benchmark Engine

This repository contains the released MedFlowBench benchmark runtime for
radiology and pathology. It includes 3D Slicer agents, pure Python radiology
baselines, QuPath agents, structured prompts, deterministic scoring, public
data helpers, and reproduction launchers.

## Tracks

| Paper track | Runtime | Released implementation |
| --- | --- | --- |
| Track A | 3D Slicer | Core viewer interaction, evidence capture, and task answering |
| Track B | 3D Slicer | The Track A radiology runners with optional analysis toolpacks |
| Track C | Python only | Sampled slices, compressed figures, M3D, or RadFM |

Track A and Track B intentionally share the same Slicer runners. Track B
enables optional segmentation, registration, MONAI Label, or VISTA3D
toolpacks; it is not a separate implementation. Track C never starts or
connects to 3D Slicer.

Pathology is organized separately and does not use Track A/B/C labels.

## Repository Layout

```text
radiology/
  slicer_track_a_b/   Slicer runners, bridge, setup, and Table 1 launcher
  track_c/            Pure Python radiology baselines
  model_serving/      Local vLLM setup and serving
pathology/            QuPath pathology benchmarks, examples, and sidecar
engines/              Shared agent, provider, scoring, and prompt code
shared/
  config/             Sanitized experiment configuration
  data_download/      Public dataset helpers
  prompts/            Versioned prompts loaded by the runners
  requirements/       Environment requirements and audited locks
```

## Agent Reproduction Workflow

The canonical agent workflow for both Track A and Track C is at the repository
root:

[`SKILL.md`](SKILL.md)

It tells an AI coding agent how to check the host, download or locate data,
install Python 3.10 and vLLM, install and verify 3D Slicer, configure paths and
credentials, run Track A and Track C five-unit smoke tests, launch full
experiments, resume interrupted work, and validate both sets of outputs.

For human-operated setup, see
[`shared/docs/reproduction.md`](shared/docs/reproduction.md). After
`.env.table1` is configured, the Track A Table 1 entry point is:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh \
  --models medgemma-1.5-4b-it \
  --limit 5
```

The default smoke scope is five units per dataset. Use `--limit 0` for the
full eligible cohorts, or `--model-set medgemma --limit 0` for both released
MedGemma rows.

## Open-Weight Models

Qwen3.5, Gemma 3, MedGemma, and Lingshu are configured in
[`shared/config/table1_models.json`](shared/config/table1_models.json). They
use the same OpenAI-compatible local vLLM path as the benchmark host. Local
weight paths and provider credentials are supplied through `.env.table1`;
none are embedded in the repository.

## Track B And Track C

Track B uses the shared Slicer runners with optional
`--segmentation_toolpacks`, `--registration_toolpacks`,
`--monai_label_toolpacks`, `--vista`, and `--activate_toolpacks` flags. The
advanced command matrix is in
[`shared/docs/reproduction.md`](shared/docs/reproduction.md).

Track C has its own one-command launcher and does not require Slicer,
WebServer, Xvfb, or Slicer extensions:

```bash
cp shared/config/track_c.env.example .env.track_c
bash radiology/track_c/scripts/reproduce_track_c.sh \
  --model medgemma-1.5-4b-it \
  --method compressed_figures \
  --limit 5
```

## Prompts And Scoring

Static model-facing prompts are versioned JSON files under
[`shared/prompts/`](shared/prompts/) and are loaded directly by the runners.
Each run records prompt identity, rendered input, model identity, traces,
evidence, and deterministic task, evidence, and localization scores. Hidden
references used for scoring are not exposed to the model.

## Data And Pathology

Datasets, weights, credentials, hidden references, and generated results are
not included. Public radiology download commands and expected layouts are in
[`shared/data_download/README.md`](shared/data_download/README.md).

The pathology module uses QuPath for BRACS and CAMELYON17 and is separate from
the radiology code. See
[`shared/docs/pathology.md`](shared/docs/pathology.md).

## Documentation

| Topic | File |
| --- | --- |
| Agent workflow for Track A and Track C | [`SKILL.md`](SKILL.md) |
| Human reproduction guide | [`shared/docs/reproduction.md`](shared/docs/reproduction.md) |
| Environments and Slicer | [`shared/docs/environments.md`](shared/docs/environments.md) |
| Open-weight models and vLLM | [`shared/docs/open_weight_models.md`](shared/docs/open_weight_models.md) |
| MedGemma Table 1 | [`shared/docs/medgemma_table1.md`](shared/docs/medgemma_table1.md) |
| Track C | [`shared/docs/track_c.md`](shared/docs/track_c.md) |
| Prompt registry | [`shared/prompts/README.md`](shared/prompts/README.md) |
| Pathology with QuPath | [`shared/docs/pathology.md`](shared/docs/pathology.md) |

## Security And License

Keep Slicer WebServer and vLLM on localhost or a protected network. Never
commit credentials, patient data, weights, runs, logs, or rendered evidence.

The benchmark code is released under Apache-2.0; see [LICENSE](LICENSE).
3D Slicer, QuPath, extensions, datasets, and model weights retain their own
licenses and terms.
