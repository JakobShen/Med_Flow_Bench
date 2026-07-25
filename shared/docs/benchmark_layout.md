# Benchmark Entry Points

Install the package and run benchmark commands from the repository root.

- `radiology/slicer_track_a_b/`: shared 3D Slicer Track A/B runners.
- `radiology/track_c/`: runtime-agnostic radiology runners and isolated
  M3D/RadFM workers.
- `pathology/benchmarks/`: QuPath pathology runners for BRACS and CAMELYON17.

Track A and Track B radiology experiments intentionally use the same three
Slicer runners. Optional toolpack flags select the Track B condition.
Track C is a sibling radiology implementation and has no 3D Slicer runtime
dependency.

See `shared/docs/reproduction.md` for complete commands.
