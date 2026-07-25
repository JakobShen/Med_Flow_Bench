# Runtime Bridges

- `radiology/slicer_track_a_b/runtime/slicer_bridge/` contains the fixed handlers loaded inside
  3D Slicer.
- `pathology/runtime/qupath_sidecar/` contains the optional Midscene GUI-control
  process.

The primary QuPath direct-control bridge is packaged under
`engines/qupath_agent_engine/software/qupath/`; the sidecar is optional.
