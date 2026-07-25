# Dependency Sets

- `base.txt`: shared radiology runtime and scoring dependencies.
- `agent.txt`: model-provider SDKs.
- `track_a_b.txt`: complete host-side dependency set for the shared Slicer
  Track A/B runners.
- `pathology.txt`: base, provider, and QuPath pathology dependencies.
- `track_c.txt`: standalone runtime-agnostic radiology dependencies.
- `vllm.txt`: Linux GPU server for Qwen, Gemma, MedGemma, and Lingshu.
- `dev.txt`: pinned test dependency used by the release checks.
- `mcp.txt`: Slicer MCP server dependencies.
- `vista3d-server.txt`: VISTA3D/MONAI service dependencies, installed after a
  CUDA-matched PyTorch build.
- `snapshots/`: sanitized package inventories from the mutable server
  environments. They are audit records, not recommended install inputs.
- `locks/vllm-linux-py310-cu13-20260725.txt`: the exact 196-package
  Python 3.10.20 environment validated for the MedGemma limit-5 clean-room
  reproduction. The vLLM installer uses this lock by default.

Create a supported environment with:

```bash
bash shared/setup/create_python_env.sh track_a_b
bash shared/setup/create_python_env.sh track_c
bash shared/setup/create_python_env.sh pathology
```

The complete environment matrix and external-runtime requirements are in
`shared/docs/environments.md`.
