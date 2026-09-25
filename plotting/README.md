# Plotting

All figure-producing scripts live here, grouped by what they plot. Every script
resolves its inputs from the repo root, so it can be run from any working
directory, and writes figures (plus the CSV summaries it derives) under
`plotting/figures/<group>/`.

- `pinder_pair/` — PINDER paper figures: performance vs throughput across
  methods and seeds, graph+mesh noise sweep, alpha-noise sweep, the two noise families
  overlaid on twin x-axes (`plot_noise_combined_common.py`), validity vs time.
  Reads the per-system result dumps in `alphasurf/tasks/pinder_pair/`.
  Every AUROC figure is computed on the system set in `common_systems.py`
  (1835 holo, 309 apo, 1582 af2, read from
  `alphasurf/tasks/pinder_pair/repaired_common_ids_20260907/`); load ids with
  `common_ids(setting)` and per-system dumps with `read_results(path, setting)`,
  which rejects any run that does not cover the set.
- `masif_ligand/` — MaSIF-Ligand figures: performance vs throughput (alone and
  combined with PINDER), standalone legends, surface-generation benchmark bars.
  The combined panels read the PINDER summary CSVs from
  `figures/pinder_pair/`, so regenerate those first.
- `benchmarks/` — surface-generation timing, discard rate and operator-time
  benchmarks. The `plot_*_discard_rate.py` / `plot_*operator*.py` scripts run
  the benchmark themselves (slurm launchers next to them) and read PDBs from
  `data/pinder-pair/pdb`; the rest read CSVs from `scripts/`.
- `meshviz/` — 3D mesh renders and interactive HTML viewers (per-method surface
  grids, failing/fragmented meshes, tufting and diffusion visualizations,
  non-manifold and duplicated-vertex highlighting).

Slurm launchers in `benchmarks/` `cd` into their own directory and write logs to
`benchmarks/log/<job-name>/`.
