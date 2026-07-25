# Benchmark Prompt Registry

This directory is the canonical, executable source for every static
model-facing benchmark prompt released with MedFlowBench. The runner scripts
do not maintain a second copy of these prompts: they call
`medflow_prompts.render_prompt()` on these JSON files and fill only
case-specific or runtime-specific placeholders.

## Runner Mapping

| Runtime | Benchmark | Structured prompt |
| --- | --- | --- |
| 3D Slicer Track A/B | UCSF-PDGM | `3d_slicer/track_a_b/ucsf_pdgm.json` |
| 3D Slicer Track A/B | NSCLC-Radiogenomics | `3d_slicer/track_a_b/nsclc.json` |
| 3D Slicer Track A/B | LUMIERE | `3d_slicer/track_a_b/lumiere.json` |
| Runtime-independent Track C | UCSF-PDGM | `track_c/ucsf_pdgm.json` |
| Runtime-independent Track C | NSCLC-Radiogenomics | `track_c/nsclc.json` |
| Runtime-independent Track C | LUMIERE | `track_c/lumiere.json` |
| Pathology with QuPath | BRACS | `pathology/common.json` + `pathology/bracs.json` |
| Pathology with QuPath | CAMELYON17 | `pathology/common.json` + `pathology/camelyon17.json` |
| Shared Slicer agent protocol | all Track A/B runs | `3d_slicer/runtime.json` |
| Optional Slicer toolpacks | Track B conditions | `3d_slicer/toolpacks.json` |

Each JSON file contains:

- a stable `prompt_id`;
- a semantic `version`;
- benchmark and track names;
- the repository `entrypoints` that consume it;
- `templates`, including conditional fragments and output contracts;
- explicit `{{placeholder}}` variables for case-specific values.

The schema is `schema.json`. Nested template objects are supported so
conditional wording such as baseline/manual/auto tool exposure,
registered/unregistered LUMIERE inputs, attached images/external workers, and
enabled/disabled code execution remains in JSON. Python selects the variant
key; it does not reproduce the wording.

## Direct Runtime Use

The loader is implemented in `engines/medflow_prompts/registry.py`. A typical
runner does:

```python
PROMPT_REF = "3d_slicer/track_a_b/ucsf_pdgm.json"

instructions = render_prompt(
    PROMPT_REF,
    "templates.system",
    answer_set="A|B|C",
    tool_guidance=render_prompt(
        PROMPT_REF,
        "templates.fragments.tool_guidance.none",
    ),
    force_gemini_guidance="",
    code_execution_guidance="",
)
```

`MEDFLOW_PROMPT_DIR` may point to a replacement prompt tree for controlled
experiments. Otherwise the loader resolves `shared/prompts/` from the source
checkout. References must be safe relative JSON paths and cannot escape the
prompt root.

## Prompt Provenance

Every model run records:

- prompt ID, semantic version, schema version, relative file reference, and
  canonical JSON SHA256;
- the fully rendered system instruction and user message;
- conditional prompt definitions used by the run.

Track A/B logs include the dataset prompt, optional-toolpack prompt, and shared
runtime prompt. QuPath logs include the common host prompt and the
benchmark-specific prompt. Track C logs include the corresponding
runtime-independent benchmark prompt.

Case paths, scene inventories, selected series, image attachments, case IDs,
timepoints, and tool budgets remain runtime values. Tool names, JSON argument
schemas, and descriptions remain in the audited tool registry because they are
API contracts rather than benchmark task prompts.

## Verification

Run the prompt-only checks without Slicer, QuPath, data, model weights, or API
credentials:

```bash
python -m pytest shared/tests/test_prompt_registry.py
```

The test recursively loads and renders every template leaf, validates
metadata, checks entrypoint paths, confirms that each prompt reference appears
in executable source, and rejects unsafe prompt paths.
