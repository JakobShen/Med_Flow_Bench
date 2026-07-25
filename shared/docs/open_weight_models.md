# Open-Weight Models Through vLLM

## Supported path

The radiology runners use a provider-neutral model boundary. The
`local_vllm` provider connects the OpenAI Python client to a local
OpenAI-compatible vLLM server:

```text
Track A or API-backed Track C runner
  -> local_vllm provider
  -> OpenAI Responses API client
  -> http://127.0.0.1:8000/v1
  -> vLLM
  -> Qwen, Gemma, MedGemma, or Lingshu weights
```

This is the public version of the path used by the earlier Qwen launchers.
Those launchers set `QWEN_BASE_URL` to a local vLLM endpoint and passed
`--provider qwen`. The release now uses the explicit model-neutral
`--provider local_vllm`, so Qwen-specific hosted API settings are no longer
mixed with local MedGemma, Gemma, or Lingshu execution.

vLLM exposes an OpenAI-compatible Responses API. During multimodal Slicer
runs, the adapter supplies tool definitions through a text protocol:

```text
<tool_call>{"name":"tool_name","arguments":{"key":"value"}}</tool_call>
```

The host validates the tool name, executes only a registered Slicer operation,
returns the JSON result and rendered images, and continues the model
conversation. This fallback does not require Qwen, MedGemma, or Lingshu to
implement provider-native function calling. The earlier Qwen launcher used
version-specific native parser flags; the public launcher omits them because
they are unnecessary here and parser names differ across vLLM releases.
Native parser flags can still be supplied through `VLLM_EXTRA_ARGS`.

Relevant implementation files:

- `engines/slicer_agent_engine/llm/registry.py`
- `engines/slicer_agent_engine/llm/adapters/openai_native.py`
- `shared/config/table1_models.json`
- `radiology/model_serving/serve_vllm.sh`

## Installation

Use Linux with a vLLM-supported accelerator and Python 3.10.x. The audited
interpreter is Python 3.10.20. 3D Slicer remains a separate application and is
not installed into this environment.

```bash
bash radiology/model_serving/create_vllm_env.sh .venv-vllm
export PYTHON_BIN="$PWD/.venv-vllm/bin/python"
export VLLM_BIN="$PWD/.venv-vllm/bin/vllm"
bash shared/setup/run_release_checks.sh --scope radiology
```

The default installer uses the complete validated lock at
`shared/requirements/locks/vllm-linux-py310-cu13-20260725.txt`. It includes
the test runner, so release checks run in the benchmark environment. Set
`VLLM_USE_LOCK=0` to resolve from `shared/requirements/vllm.txt` only when a
different accelerator requires another CUDA/PyTorch wheel stack. If the
cluster requires a fully site-specific vLLM installation, finish it with:

```bash
python -m pip install -e ".[agent,dev]"
```

## Table 1 model registry

List the open-weight rows:

```bash
python radiology/model_serving/vllm_config.py list --set open
```

| Key | Default model source | Access notes |
| --- | --- | --- |
| `qwen3.5-35b-a3b` | `Qwen/Qwen3.5-35B-A3B` | Qwen model terms |
| `qwen3.5-27b` | `Qwen/Qwen3.5-27B` | Qwen model terms |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | Qwen model terms |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | Qwen model terms |
| `gemma-3-27b-it` | `google/gemma-3-27b-it` | Accept the Gemma license |
| `medgemma-1.5-4b-it` | `google/medgemma-1.5-4b-it` | Gated Health AI terms |
| `medgemma-27b-it` | `google/medgemma-27b-it` | Gated Health AI terms |
| `lingshu-32b` | `lingshu-medical-mllm/Lingshu-32B` | Upstream model terms |

For gated repositories, accept the upstream terms and export `HF_TOKEN`.
Review every model card before redistribution or clinical use. The benchmark
is for research evaluation and does not make a clinical-use claim.

GPU requirements depend on model size, dtype, context length, tensor
parallelism, and any local quantization. The launcher therefore does not claim
one universal GPU configuration. Common overrides are:

```bash
export VLLM_TENSOR_PARALLEL_SIZE=2
export VLLM_GPU_MEMORY_UTILIZATION=0.90
export VLLM_MAX_MODEL_LEN=32768
export VLLM_MODEL_PATH=/path/to/local-or-quantized-weights
```

For a multi-model Table 1 run, use the per-model path variables documented in
`shared/config/table1.env.example`; a global `VLLM_MODEL_PATH` is intended for
a one-model run.

## Start one model

This command runs the server in the foreground:

```bash
bash radiology/model_serving/serve_vllm.sh medgemma-1.5-4b-it
```

The default endpoint is `http://127.0.0.1:8000/v1`, with API key `EMPTY`.
Override `VLLM_HOST`, `VLLM_PORT`, or `VLLM_API_KEY` when needed. Keep the
endpoint bound to localhost or a protected cluster network.

In another shell:

```bash
export LOCAL_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export LOCAL_VLLM_API_KEY=EMPTY
export LOCAL_VLLM_MAX_OUTPUT_TOKENS=2048
export LOCAL_VLLM_MAX_INPUT_IMAGES=16
export LOCAL_VLLM_MAX_HISTORY_TOOL_ROUNDS=3
export LOCAL_VLLM_MAX_TOOL_RESULT_CHARS=8000
export MODEL=google/medgemma-1.5-4b-it
```

Track A:

```bash
python radiology/slicer_track_a_b/benchmark_UCSF_PDGM.py \
  --data_root "$UCSF_DATA_ROOT" \
  --csv_path "$UCSF_METADATA_CSV" \
  --provider local_vllm \
  --model "$MODEL" \
  --max_rounds 20 \
  --limit 1
```

API-backed Track C:

```bash
python radiology/track_c/benchmark_UCSF_PDGM_track_c.py \
  --data_root "$UCSF_DATA_ROOT" \
  --csv_path "$UCSF_METADATA_CSV" \
  --method compressed_figures \
  --provider local_vllm \
  --model "$MODEL" \
  --limit 1
```

Track C does not start or communicate with 3D Slicer. It sends generated
figures directly to the same local vLLM Responses endpoint.

The complete three-dataset Track C path can manage the vLLM server itself:

```bash
cp shared/config/track_c.env.example .env.track_c
bash radiology/track_c/scripts/reproduce_track_c.sh
```

The default is `medgemma-1.5-4b-it`, the paper's `compressed_figures`
representation, and five units per dataset. No Slicer or Xvfb setting belongs
in `.env.track_c`.

## One-command Table 1 orchestration

First create the untracked experiment configuration:

```bash
cp shared/config/table1.env.example .env.table1
```

Fill the dataset, Slicer, Python, vLLM, and credential paths. The default
five-unit-per-dataset end-to-end check is one command:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh
```

For the full eligible cohorts reported in Table 1:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh --limit 0
```

The manifest default is `TABLE1_MODEL_SET=all`: six hosted rows and eight
open-weight vLLM rows. To reproduce only the open-weight block:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh --model-set open
```

To reproduce the two MedGemma rows without requesting any closed-source model:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh --model-set medgemma
```

The complete MedGemma smoke and full-run procedure is in
[`medgemma_table1.md`](medgemma_table1.md).

To run one row:

```bash
bash radiology/slicer_track_a_b/scripts/reproduce_table1.sh --models lingshu-32b
```

The script explicitly runs the paper's Track A protocol:

- LUMIERE registered image space, 139 eligible questions after the released
  evidence and label-frequency filters;
- 495 eligible UCSF-PDGM cases;
- 162 eligible NSCLC cases with AMC excluded;
- 20 tool-calling rounds;
- no segmentation, registration, MONAI Label, or VISTA3D toolpack flags.

`TABLE1_LIMIT` defaults to `5`; blank dataset-specific limits inherit it. Set
`TABLE1_LIMIT=0` or pass `--limit 0` for all eligible cases. Resume markers
are written after each model/dataset combination, and the final output includes
`table1_results.csv` and `table1_results.json`.

Hosted preview endpoints can be renamed or retired. The manifest records the
paper model names, while environment variables such as
`TABLE1_MODEL_ID_GEMINI_3_FLASH_PREVIEW` can override a current endpoint
without changing benchmark code. Exact numerical equality is not guaranteed
when a provider has changed model snapshots or serving behavior; the run
artifacts record the requested model and full execution trace.

## Upstream references

- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
- [vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Gemma 3 27B IT](https://huggingface.co/google/gemma-3-27b-it)
- [MedGemma 1.5 4B IT](https://huggingface.co/google/medgemma-1.5-4b-it)
- [MedGemma 27B IT](https://huggingface.co/google/medgemma-27b-it)
- [Lingshu-32B](https://huggingface.co/lingshu-medical-mllm/Lingshu-32B)
