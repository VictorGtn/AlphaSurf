# Tests

`pytest` from the repo root runs everything in `tests/` (see `pytest.ini`).
Set `PYTHONPATH` to include the repo root and `cgal_alpha_bindings/build_py310`
so the alpha-complex bindings are importable.

- `test_patch_operators.py`, `test_patch_graph_geometry.py`,
  `test_patch_graph_binding.py` — patch operators and the CGAL patch-graph
  binding.
- `test_backbone_mask.py`, `test_sampling.py`, `test_checkpointing.py`,
  `test_s3f_exact_batching.py`, `test_s3f_exact_preprocessing.py`,
  `test_surface_esm.py` — S3F pretraining: masking, sampling, checkpoint
  stripping, exact batching/preprocessing, surface-ESM injection.
- `test_scoring.py` — ProteinGym scoring and evaluation.
- `test_frame_modes.py` — MISATO binding-site frame selection.
- `test_curvature_ext.py` — the `cpp_curvature` extension against libigl and
  analytical curvature; needs `curvature_ext` built in `cpp_curvature/` and
  `igl` installed.

`manual/` holds harnesses that need a real environment rather than assertions,
and is excluded from collection: `test_inference.py` composes the PINDER Hydra
config, builds a random-weight checkpoint and runs embed/interact
(`test_inference.sh` is its slurm launcher, logging to `manual/log/`).

Task directories still contain `test.py` / `test_*.sh` — those are test-split
model evaluations, not unit tests, and stay with their task.
