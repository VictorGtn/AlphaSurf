# AlphaSurf

## Noting practices

When the user writes `(note this)` in a message, the FIRST action must be to update this file (or `~/.claude/CLAUDE.md` if the fact is not repo-specific) to reflect the noted information — before any other work. Then continue with the request.

This file must stay a thorough, accurate overview of codebase functionality. Update it whenever a feature is added, a task is created, or the architecture changes; that update is part of the feature, not a follow-up.

Design docs go in `notes/`.

## What the project does

AlphaSurf is a protein structure encoder that jointly encodes residue/atom **graphs** and molecular **surfaces**, generating the surface on-the-fly from a weighted alpha complex instead of requiring precomputed MSMS/NanoShaper meshes.

The surface algorithm (`docs/report.md`, `docs/pocket_detection_algorithm.md`):
- Weighted Delaunay (regular) triangulation of atom centers, weights from van der Waals radii plus a solvent probe radius (power distance).
- Alpha-shape classification of cells/facets/edges as INTERIOR / EXTERIOR / SINGULAR / REGULAR via `mu(sigma) <= alpha`, giving a naive alpha-complex surface.
- Interior-pocket removal: naive alpha shapes expose spurious patches lining trapped interior voids. A 3D cell-adjacency graph (nodes = EXTERIOR tetrahedra, edges = shared EXTERIOR facets) with union-find from the point at infinity identifies EXTERIOR components unreachable from infinity; these are reclassified as solid and the surface re-extracted. This volumetric approach replaces naive 2D mesh face clustering.

`paper/` holds the draft manuscript and figures; `docs/pinch_points.tex` covers surface pinch-point degeneracies.

## Environment

Conda env `atomsurf` (Jean Zay: `$SCRATCH/atomsurf_h100_env`). `PYTHONPATH` must include the repo root and `cgal_alpha_bindings/build*`.

## Layout

- `alphasurf/` — the installable package (see below).
- `scripts/` — one-off analysis and benchmarking scripts plus their slurm launchers (surface-generation benchmarks, MaSIF/PINDER throughput, connected-component and pocket-detection sweeps, PDB preprocessing, spectral comparison). Not part of the package API.
- `tests/` — pytest unit tests; `tests/manual/` holds environment-dependent harnesses, excluded from collection. See `tests/README.md`.
- `plotting/` — figure scripts grouped by subject (`pinder_pair/`, `masif_ligand/`, `benchmarks/`, `meshviz/`), writing to `plotting/figures/<group>/`. Every PINDER AUROC figure is computed on the frozen common system set in `plotting/pinder_pair/common_systems.py` (1835 holo, 309 apo, 1582 af2, ids in `alphasurf/tasks/pinder_pair/repaired_common_ids_20260907/`): call `common_ids(setting)` and `read_results(path, setting)`, which rejects a run that does not cover the set. See `plotting/README.md`.
- `docs/` — algorithm and design documentation.
- `cgal_alpha_bindings/` — C++/pybind11 CGAL sources for alpha-complex extraction.
- `cpp_curvature/` — C++/pybind11 extension for principal curvatures and vertex normals.
- `bin/` — vendored MSMS binaries (`msms_linux`, `msms_macos`, `msms_windows`).
- `data/` — datasets and cached surfaces, populated locally, not code.
- `tasks/` — legacy leftover directory; real task implementations live in `alphasurf/tasks/`.

## Package structure

- `alphasurf/protein/` — data pipeline. `protein.py` / `main_data.py` (data structures), `graphs.py` / `residue_graph.py` / `atom_graph.py` (graph construction, PDB parsing, atom radius tables), `surfaces.py` (`SurfaceObject` / `SurfaceBatch`, PyG `Data`/`Batch` subclasses holding verts/faces/operators/HKS), `create_surface.py` (surface generation), `create_operators.py` / `patch_operators.py` (DiffusionNet-style Laplacian and gradient operators, spherical patch graphs), `features.py`, `create_esm.py`, `transforms.py`, `protein_loader.py`. See `alphasurf/protein/README.md`.
- `alphasurf/networks/` — top-level architecture: `protein_encoder.py` (joint graph+surface encoder used by every task), `input_feat_encoder.py`.
- `alphasurf/network_utils/` — building blocks. `communication/` (graph-surface message passing), `misc_arch/` (`dgcnn.py`, `dmasif_encoder.py`, `pronet.py`, `gvp_gnn.py`, `pointnet.py`, `poissonnet.py`, `deltaconv.py`, `gatr_encoder.py`, `s3f_blocks.py`).
- `alphasurf/utils/` — `config_utils.py` (Hydra helpers), `data_utils.py`, `torch_utils.py`, `learning_utils.py`, `metrics.py`, `callbacks.py` (Lightning callbacks, wandb wiring), `batch_sampler.py`, `timing_stats.py`, `wrappers.py`, `python_utils.py`, `atom_utils.py`.
- `alphasurf/tasks/` — one self-contained directory per downstream task, each with `train.py`, `conf/`, model, datamodule, preprocessing and slurm scripts.
- `alphasurf/tasks/shared_conf/` — Hydra fragments shared across tasks (`config.yml` base hyperparameters, `blocks_zoo*.yaml`, `model_global_variables.yaml`, `encoder/`, `optimizer/`, `scheduler/`).

## Surface generation

`alphasurf/protein/create_surface.py` dispatches on `surface_method`:
- `alpha_complex` → `pdb_to_alpha_complex()` → `cgal_alpha_algo2.compute_alpha_complex_algo2_from_atoms` (the project's own method).
- `msms` → `pdb_to_msms()`, wraps the vendored MSMS binary, parses `.vert`/`.face`.
- `edtsurf` → `pdb_to_edtsurf()`, expects `<parent>/EDTSurf/EDTSurf`.
- `nanoshaper` → `pdb_to_nanoshaper()`, expects `<parent>/nanoshaper-master/NanoShaper`, produces SES surfaces.
- `patch_graph` → `ProteinLoader._generate_patch_graph_surface()`, one node per exposed spherical patch from `cgal_patch_graph`, operators from `patch_operators.build_patch_operators`. Per-node geometry features are log patch area, exposed sphere fraction, log boundary-to-area ratio, arc-length-weighted signed junction angle between neighboring patch normals (negative = concave), 16 HKS channels and the patch normal, plus the parent atom's chemistry: element one-hot (12), residue hydrophobicity and, when the PDB carries charges, partial charge (the input width is inferred from data). Cached graphs live in `data/masif_ligand/patch_graphs_exact_a<alpha>_p<probe>/`, written by `tasks/masif_ligand_new/precompute_patch_graphs.py`. Train with `tasks/masif_ligand_new/train_patch_graph{,_h100}.sh`, with `diffusion_net.with_gradient_features=true` (default). Evaluate a checkpoint with `CKPT=<path> SEED=<seed> sbatch test_patch_graph.sh`. Three-seed result and the history of the gradient-feature collapse are in `notes/patch_gradient_collapse.md`.

Post-processing clusters triangles to drop disconnected fragments (`cluster_triangles_by_vertex_sharing`; vertex-sharing for alpha_complex, edge-sharing for msms) and repairs the mesh. Alpha-complex surfaces additionally require the `cpp_curvature` extension for curvatures and normals.

`alphasurf/protein/protein_loader.py` provides the unified `ProteinLoader` for disk-cached and on-the-fly modes, with a fixed transform order: PDB parse, atom noise, mesh generation, patch extraction, mesh noise, operator computation, graph build, feature expansion. Surfaces serialize to `.pt` (also `.npz`).

Cached data lives under `data/`: `data/pinder-pair{,-all,-inf}/`, `data/masif_ligand/`, `data/misato/`, `data/cath/`, `data/surfaces_full_msms_1.0_False/`, `data/example_files/` (fixtures used by module `__main__` blocks).

## Native extensions

- `cgal_alpha_bindings/` — CMake + pybind11, one module per `.cpp`: `cgal_alpha`, `cgal_alpha_algo2` (primary), `cgal_alpha_raw`, `cgal_alpha_tagged`, `cgal_pmp_repair`, `cgal_alpha_edge_analysis`, `cgal_patch_graph` (spherical patch graphs via SBL's exact spherical kernel, feeds `patch_operators`). Build: `mkdir build && cd build && cmake .. && make cgal_alpha_algo2 -jN`. Needs CGAL 5.x+, GMP, MPFR, pybind11, CMake 3.16+, Python 3.10+. Only `cgal_alpha_algo2` is in the default target; the rest are `EXCLUDE_FROM_ALL` and are built by name. `cgal_alpha` and `cgal_patch_graph` are declared only when `SBL_ROOT` points at an SBL checkout. The `.so` lands in `cgal_alpha_bindings/build/` and is auto-discovered relative to the source tree, but slurm workers using `spawn`/`forkserver` need `CGAL_BINDINGS_DIR` and `PYTHONPATH` set explicitly. `build_py310/` is a Python-version-specific build tree.
- `cpp_curvature/` — single-file pybind11 extension built with `python build.py` (raw `g++ -O3 -shared -std=c++17 -fPIC`, Eigen3 include auto-discovered from the conda prefix, `/usr/include/eigen3`, or the venv). Equivalent to `igl.per_vertex_normals` but takes custom vertex normals. The `.so` is a build artifact and is not tracked; build it before running the tests.

## Tasks

- `pinder_pair/` — protein-protein interaction on PINDER. Predicts per-residue-pair interaction probability and per-residue binding-site scores. `dataset.py` (pair loading, negative-pair generation), `datamodule.py`, `model.py`, `pl_model.py` (`PinderPairModule`), `train.py` / `test.py` / `precompute.py`, launchers `train_onfly_h100.sh`, `train_disk_h100.sh`. Three test settings (holo/apo/af2) and `on_fly.noise_mode=joint_mesh` augmentation. A local trained checkpoint sits at `alphasurf/tasks/pinder_pair/ckpt/last.ckpt`; `*.ckpt` is gitignored and no weights are published. Noise and throughput sweep artifacts are checked in alongside the code.
- `masif_ligand_new/` — MaSIF-Ligand 7-class ligand classification. `dataset.py` (`BaseProteinDataset`, the reusable protein-loading base), `model.py` (`MasifLigandNet`: encoder, k-NN pooling around ligand coordinates, MLP head), `pl_model.py`, `datamodule.py`, `train.py`. `masif_ligand/` is the superseded implementation.
- `s3f_pretrain/` — S3F-style self-supervised pretraining on CATH. `dataset.py`, `dataset_s3f_exact.py`, `sampling.py`, `precompute_s3f_exact.py`, `checkpointing.py`, `download_cath.sh`. Produces the checkpoints used by `proteingym`.
  Surface source follows the repo-wide `on_fly` convention: leave `on_fly` null to read precomputed clouds from `precompute_dir` (`CATHDatasetS3FExact`), or set it to generate them at runtime (`CATHDatasetS3FExactOnFly`). With `on_fly.surface_in_workers` (default true) each worker builds its protein's cloud on CPU in `__getitem__`; set it false to build the whole batch at once on the GPU in `datamodule.on_after_batch_transfer` via `dataset_s3f_exact.attach_surfaces`, which needs `KEOPS_CUDA=1` at submit time so the launcher loads the cuda module. Both modes get their geometry from `precompute_s3f_exact.build_s3f_surfaces`.
  `build_s3f_surfaces` calls `s3f_official/surface.py`, a verbatim copy of upstream `s3f/surface.py` at commit `2efab6a`, so it produces the same features as `script/process_surface.py`. Refresh that copy with `cp /cluster/CBIO/data2/vgertner/S3F_official/s3f/surface.py s3f_official/surface.py`.
  Regenerate `data/cath/s3f_exact_precomputed` before training on it: the surfaces there were written with the earlier in-repo dMaSIF and curvature copies (`reg=0.01` rather than upstream's `1e-10`), so they do not match the current code.
  `precompute_alpha.py` precomputes alpha-complex surfaces and graphs for CATH, replicating `tasks/pinder_pair/precompute.py`: one task per protein on a spawned pool with a stall watchdog. Run it through `precompute_alpha_jz.sh`. `time_s3f_official_preprocess.py` and `timing_harness.py` time the upstream S3F pipeline the same way, through `time_s3f_official_jz.sh`. Both report `Throughput: X proteins/s` over the whole wall clock, directly comparable to the PINDER precompute numbers. Set `LIMIT` to cap the protein count and `NUM_WORKERS` for the pool size.
  `convert_s3f_official_surfaces.py` repackages the upstream pickles written by `time_s3f_official_preprocess.py` into the `.pt` format read by `encoder=s3f_exact` (`data/cath/dmasif_s3f_precomputed`); run it through `convert_s3f_official_surfaces_jz.sh` (`PKL_DIR`, `OUTPUT_DIR`). `precompute_alpha_s3f.py` builds the same format from backbone-only alpha complexes (`data/cath/alpha_s3f_precomputed`); add `--laplacian cotan` to take the HKS eigenbasis from the cotan Laplacian of `compute_operators` (`create_operators.laplacian_eigenbasis`) on the alpha mesh, with S3F's eigenpair count, instead of S3F's point-cloud Laplacian (`data/cath/alpha_s3f_cotan_precomputed`). Train either with `RUN_NAME=<name> PRECOMPUTE_DIR=<dir> sbatch train_s3f_exact_jz.sh`.
  Measured on one H100 with 20 workers over full CATH (31,565 proteins): alpha complex 76.44 proteins/s, upstream S3F 2.80 proteins/s, where `compute_eigens` is 95% of the S3F cost.
  The `s3f_exact` encoder (`shared_conf/encoder/s3f_exact.yaml`, blocks in `network_utils/misc_arch/s3f_blocks.py`) reimplements the released S3F inference forward pass. `tests/manual/test_s3f_official_parity.py` checks it numerically against the upstream clone at `/cluster/CBIO/data2/vgertner/S3F_official` (commit `2efab6a`), loading official weights into our blocks; it must be run with the `atomsurf` env, which has `torchdrug`, `rdkit==2022.9.5`, `pykeops` and `robust_laplacian`. `S3FFusion.readout` selects `released` (one batch-wide surface mean broadcast to all residues, which is what the released code and hence the published S3F numbers do) or `local` (the paper's per-residue 60-NN pool). Only `released` matches upstream.
- `proteingym/` — zero-shot fitness prediction on the ProteinGym substitutions benchmark (217 DMS assays), reproducing S3F's protocol. `scoring.py` / `evaluate.py` score masked mutant-vs-WT log-odds, with an ESM-2 fallback for low-pLDDT AF2 residues; graph and surface are regenerated per masked-position set. `s3f_exact` checkpoints are scored on precomputed S3F surfaces of the AF2 structures instead: build them by running `precompute_alpha_s3f_jz.sh --pdb-dir <af2 dir>` (alpha, `data/proteingym/s3f_alpha_surfaces`) or `time_s3f_official_jz.sh --pdb-dir <af2 dir> --output-dir <pkl dir>` followed by `convert_s3f_official_surfaces_jz.sh` with `PKL_DIR`/`PDB_DIR`/`OUTPUT_DIR` (dMaSIF, `data/proteingym/s3f_dmasif_surfaces`), then evaluate with `CKPT=<ckpt> S3F_SURFACE_DIR=<surface dir matching the training cache> sbatch evaluate_jz_h100.sh`. A full `s3f_exact` eval takes ~2 h 15; to fit the 2 h dev QoS, submit it as two jobs over the runtime-balanced assay halves in `tasks/proteingym/runs/splits/s3f_half{0,1}.txt` with `ASSAY_IDS="$(cat runs/splits/s3f_half<i>.txt)" OUTPUT_DIR=<run>_half<i> sbatch --qos=qos_gpu_h100-dev --time=01:40:00 evaluate_jz_h100.sh`, then concatenate the two `summary.csv` files before aggregating. `summary.csv` records, per assay, how much of it was scored structurally (`num_scored`, `num_groups_geometry_failed`, `num_positions_low_plddt`) — always read those next to the Spearman. See `alphasurf/tasks/proteingym/README.md` and `notes/proteingym_gap_analysis.md` for why AlphaSurf trails S3F.
- `misato_binding_site/` — residue-level binding-site prediction on MISATO. Label: C-alpha within 10 A of any ligand heavy atom at frame 0; official BlastP 30%-identity complex-level splits; trains on random MD frames, evaluates on frame 0; the model never sees the ligand. `preprocess.py` writes `binding_site/<pdb_id>.pt`. `evaluate_guo_batch64.py` implements the Guo et al. factorized batch-64 residue-pooled evaluation; `tune_threshold.py` tunes the decision threshold. See its `README.md`.
- `misato_affinity/` — MISATO binding-affinity prediction. `build_affinity.py`, `dataset.py`, `model.py`, `pl_model.py`, `preprocess.py`, `train.py`.
- `inference/` — `embed.py` loads a `PinderPairModule` checkpoint plus a raw PDB and dumps `graph_embedding`, `surface_embedding`, `graph_node_pos`, `surface_verts` to a `.pt`. Run as `python embed.py --ckpt <path> --pdb protein.pdb`.

## Running things

Config is Hydra + OmegaConf (`hydra-core==1.3.2`, `omegaconf==2.3.0`). Each task has `conf/config.yaml` whose `hydra.searchpath` is `pkg://alphasurf.tasks.shared_conf`, so it resolves wherever the package is importable. `data_dir` is mandatory (`???`) in every task config: pass it on the command line. Paths to files inside the package are written `${alphasurf_dir:<relative path>}`, a resolver registered in `alphasurf/__init__.py`.

Runs are launched from inside the task directory with Hydra `key=value` overrides:

```
cd alphasurf/tasks/pinder_pair
python train.py data_dir=... on_fly.surface_method=alpha_complex encoder=pronet_gvpencoder.yaml epochs=500 loader.batch_size=4
```

Disk mode is `on_fly=null`, preceded by `python precompute.py data_dir=...`. On-the-fly surface parameters live under `on_fly.*` (`surface_method`, `alpha_value`, `face_reduction_rate`, `noise_mode`). Some MISATO entrypoints run as modules: `python -m alphasurf.tasks.misato_binding_site.preprocess --data-dir ...`.

Scheduler gotcha: `get_lr_scheduler` in `alphasurf/utils/learning_utils.py`
derives the decay length from `cfg.epochs - warmup_epochs` and ignores
`cfg.scheduler.T_max`, and `configure_optimizers` registers the scheduler with
`interval: "epoch"`. A cosine run therefore anneals over epochs, not steps, and
a short-epoch-count run ends at `eta_min` (1e-8). Resuming such a checkpoint for
more epochs requires patching the saved optimizer/scheduler state, since
Lightning restores the scheduler `state_dict` (including `T_max`) over whatever
the new config builds.

Lightning writes TensorBoard logs to `cfg.log_dir` (default `./`) and checkpoints to `<tb_logger.log_dir>/checkpoints/`; resume also scans the log dir for `hpc_ckpt_*.ckpt` (slurm preemption checkpoints). wandb is opt-in via `cfg.use_wandb` (default `False`), wired through `alphasurf.utils.callbacks.add_wandb_logger` with `cfg.project_name` / `cfg.run_name`; slurm scripts set `WANDB_MODE=offline` and `WANDB_DIR=$(pwd)/wandb_logs`, synced later with `wandb_sync_jz.sh`.

Tests: `pytest` from the repo root (`pytest.ini`: `testpaths = tests`, `norecursedirs = tests/manual`). Alpha-complex tests need `cgal_alpha_bindings/build_py310` on `PYTHONPATH`; `test_curvature_ext.py` needs `curvature_ext` built and `igl` installed. `test.py` / `test_*.sh` inside task directories are model evaluation scripts, not unit tests.

## Script outputs

Everything a script in `scripts/` produces goes under `scripts/outputs/<experiment>/` — raw CSV, summary CSV and figures for one experiment live in the same directory. Never write results to the root of `scripts/`. Existing directories: `masif_benchmark/`, `pinder_benchmark/`, `discard_rate/`, `cc_analysis/`, `interface_cc/`, `msms_coverage/`, `operator_timing/`, `surface_sizes/`, `timing/`, `meshviz/`, `mesh_statistics/`, `pinder_mesh_statistics/`, `surface_duplicate_cc/`, `s3f_esm_baseline/`, `patch_gradient_diag/` (mesh-vs-patch operator statistics, trained-checkpoint activation probes and ProNet init checks from `scripts/patch_gradient_*.py` and `scripts/patch_pronet_init_check.py`, launched with `scripts/pronet_init_check.sh`), the `*_serial*` benchmark dumps, and `diffusion_viz/`.

`.gitignore` ignores `**/outputs/`, so nothing under `scripts/outputs/` is tracked; all of it is regenerable. Readers in `plotting/` must therefore reference `scripts/outputs/<experiment>/`, not `scripts/`. The pre-existing `scripts/cc_sweep_output*/`, `failure_plots/`, `heat/` and `viz/` dumps stay outside `outputs/` (they are tens of GB and have their own ignore rules; the summary CSVs inside `cc_sweep_output*` remain tracked).

New scripts default `--output-dir` to `<script dir>/outputs/<experiment>` rather than the script directory.

## Spectral comparison figures

`scripts/spectral_comparison.py` computes the heat-diffusion agreement between alpha-complex and MSMS surfaces: Laplacian eigenpairs, HKS, the atom heat coupling `H_t(i, j)` and geodesic distances, on the exact atom correspondence (MSMS reports the sphere each vertex sits on, read through `create_surface.parse_verts(keep_atom_idx=True)`; alpha-complex vertices are atom centres). It writes `spectral_comparison.csv` plus one `.npz` per protein and pair, by default under `scripts/outputs/spectral_heat/`. Run it through `spectral_comparison.sh` (`PDB_DIR` required), which also draws the distributions when `PAIRS` is left at its default.

`plotting/spectral/` reads that run and draws the paper figures into `plotting/figures/spectral/`. Each of the three adds `scripts/` to `sys.path` to import `spectral_comparison`:
- `plot_spectral_distributions.py` — violins per metric and diffusion time, worst-tail curves, kernel correlation per geodesic distance window, and the per-sample HKS scatter.
- `plot_kernel_profile.py` — the appendix kernel fall-off figure.
- `plot_dirac_diffusion.py` — the Dirac-diffusion figure; it also imports `mesh_collection`, `mesh_limits`, `MESH_EDGE_COLOR` and `shaded_face_colors` from `plotting/meshviz/visualize_all_methods.py`, so the panels match the surface-grid figure. `--render pymol` needs PyMOL; `--render mesh` and `vector` do not.

The `sas` and `sas_dec` mesh kinds come from `cgal_sbl_sampling`, which needs `SBL_ROOT` at build time. Every other kind works without SBL.

## Slurm log files

Every folder with slurm scripts (`scripts/`, `plotting/benchmarks/`, `tests/manual/`, `alphasurf/tasks/{pinder_pair,masif_ligand_new,inference,proteingym}/`) has a `log/` tree. Each script's `#SBATCH --output` / `--error` writes to `log/<job-name>/%x_%j.{log,err}` (a few scripts use `.out`; intentional `/dev/null` is left alone).

When introducing a new `--job-name`, `mkdir -p log/<new-name>` before the first `sbatch` invocation — SBATCH parses `--output`/`--error` before the script body runs, so slurm will fail to open the output file if the directory doesn't exist.

## Submitting jobs on Jean Zay

Launchers rely on `module`, a bash function inherited from the submitting environment, so submit from a login shell with `bash -lc` and pass run parameters as `VAR=value` prefixes:

```
ssh jean-zay 'bash -lc "cd <task dir> && CKPT=... SCORING_METHOD=... sbatch <script>.sh"'
```

Check state with `sacct -j <id> --format=JobID,JobName%22,State,ExitCode,Elapsed`.

## Reminder

`(note this)` in a user message means: update this file first, then do the work.
