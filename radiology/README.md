# Radiology Tracks

The radiology benchmark has two separate execution boundaries:

| Directory | Paper track | Requires 3D Slicer | Purpose |
| --- | --- | --- | --- |
| `slicer_track_a_b/` | Track A and Track B | Yes | Viewer-native agent runs, optional toolpacks, Slicer bridge, evidence capture, and Table 1 orchestration |
| `track_c/` | Track C | No | Sampled slices, compressed figures, API inference, M3D/RadFM workers, and deterministic scoring |
| `model_serving/` | Shared | No | Local OpenAI-compatible vLLM serving for supported open-weight models |

Track A and Track B intentionally share the same three Slicer runners. Track B
is selected by enabling optional segmentation, registration, MONAI Label, or
VISTA3D toolpacks.

Track C is not implemented through 3D Slicer. Its runners do not import the
Slicer client, Slicer launcher, bridge, MRML runtime, or Slicer application
bootstrap. They can run in the standalone `track_c` Python environment:

```bash
bash shared/setup/create_python_env.sh track_c
source .venv-track_c/bin/activate
python radiology/track_c/benchmark_UCSF_PDGM_track_c.py --help
```

Shared dataset parsing, provider adapters, session logging, and deterministic
scoring are ordinary host-side Python modules and do not require a medical
viewer.
