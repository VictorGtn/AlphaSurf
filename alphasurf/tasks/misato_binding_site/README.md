# MISATO binding-site prediction

This task reproduces the target and split protocol used by Guo et al., while
using AlphaSurf as the residue encoder:

- official MISATO BlastP 30%-identity train/validation/test splits;
- one node per protein C-alpha;
- positive label when the C-alpha is within 10 Å of any stored ligand atom;
- a uniformly sampled MD frame during training and trajectory frame 0 for
  validation/test;
- fixed labels computed once from raw trajectory frame 0 (the reference
  conformation), with hydrogens removed as in Guo et al.'s processed input;
- no adaptability, correlations, or other trajectory-derived summary features;
- unweighted two-class cross-entropy;
- Guo-style factorized batch-64 evaluation.

The raw MISATO HDF5 file contains `trajectory_coordinates` directly, so the
loader reads a single frame lazily and does not require the separate
trajectory/topology archive. The ligand is never included in the input.

```bash
# Resume MD.hdf5 and fetch official split lists (login/download node).
bash alphasurf/tasks/misato_binding_site/download_misato.sh

# Extract compact per-complex files, then train.
sbatch alphasurf/tasks/misato_binding_site/preprocess.sh
sbatch alphasurf/tasks/misato_binding_site/train.sh

# Summarize protein sizes by official split (20 CPU workers).
sbatch alphasurf/tasks/misato_binding_site/analyze_sizes.sh

# Jean Zay H100: defaults to 200 epochs and batch size 64.
sbatch alphasurf/tasks/misato_binding_site/train_jz.sh
```

The split is always applied at complex level: all frames belonging to a complex
remain in the same train, validation, or test partition.

## First-frame evaluation

[evaluate_guo_batch64.py](evaluate_guo_batch64.py) implements the factorized
batch-64 aggregation used by Guo et al. It preserves test-split order, groups
systems into chunks of 64, pools residue predictions within each chunk, and
reports the residue-count-weighted mean across chunks. F1 uses a threshold
selected on frame-0 validation predictions and frozen for frame-0 testing;
AUROC and AUPRC do not use a fixed threshold.

```bash
sbatch alphasurf/tasks/misato_binding_site/evaluate_guo_batch64_jz.sh \
  /path/to/seed1.ckpt \
  /path/to/seed2.ckpt \
  /path/to/seed3.ckpt
```
