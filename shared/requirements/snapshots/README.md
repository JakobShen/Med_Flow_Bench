# Server Environment Snapshots

These files record the packages present on 2026-07-24 in the three mutable
server environment roles used during development:

- `track_a_b-server-20260724.txt`: Track A/B benchmark host
- `track_c-server-20260724.txt`: Track C benchmark host
- `vista3d-server-20260724.txt`: VISTA3D service

All three were Conda-prefix environments using Python 3.10.20. All personal
paths have been removed, and the editable benchmark package is omitted.

These are audit snapshots, not recommended installation files and not
historical locks of every paper run. The Track A/B and Track C environments had
234 and 232 packages respectively and both contained vLLM, Torch, MCP, and
HTTP-server packages because environments were reused during development.
Installing those snapshots would unnecessarily couple the benchmark host to a
specific CUDA stack.

Use the curated files one directory above for reproduction. Use
`vllm.txt` only on the model-serving machine and use `vista3d-server.txt` only
for the VISTA3D service.
