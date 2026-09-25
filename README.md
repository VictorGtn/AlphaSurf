# AlphaSurf

Implementation of AlphaSurf.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
    - [Environment setup](#environment-setup)
    - [CGAL alpha complex bindings](#cgal-alpha-complex-bindings)
    - [Curvature extension](#curvature-extension)
- [Datasets](#datasets)
- [Tasks](#tasks)
    - [MasifLigand](#masifligand)
    - [PINDER-Pair](#pinder-pair)
    - [MISATO binding-site prediction](#misato-binding-site-prediction)
    - [MISATO affinity prediction](#misato-affinity-prediction)
    - [S3F pretraining on CATH](#s3f-pretraining-on-cath)
    - [ProteinGym](#proteingym)
- [Inference](#inference)
- [Reproducing the figures](#reproducing-the-figures)

## Description

AlphaSurf is a protein structure encoder that jointly encodes graphs and surfaces, with on-the-fly alpha complex surface generation during training.

<img src="paper/surfaces.png">

## Installation

### Environment setup

```bash
conda create -n alphasurf python=3.10 -y
conda activate alphasurf
```

Install PyTorch and PyG (CUDA 11.8):

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.6.1
pip install torch_scatter torch_sparse torch_spline_conv torch_cluster -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
pip install pyg-lib==0.4.0 -f https://data.pyg.org/whl/torch-2.4.1+cu118.html
```

Install the remaining dependencies:

```bash
pip install git+https://github.com/pvnieo/diffusion-net-plus.git
pip install -r requirements.txt
```

### CGAL alpha complex bindings

On-the-fly surface generation requires CGAL Python bindings. These are located in `cgal_alpha_bindings/`.

#### Dependencies

- CGAL 5.x+
- GMP, MPFR
- pybind11
- Python 3.10+
- CMake 3.16+

#### Build

**Linux (Ubuntu/Debian)**

```bash
sudo apt install libcgal-dev libgmp-dev libmpfr-dev python3-dev cmake
pip install pybind11

cd cgal_alpha_bindings
mkdir build && cd build
cmake ..
make cgal_alpha_algo2 -j$(nproc)
```

**macOS (Homebrew)**

```bash
brew install cgal gmp mpfr
pip install pybind11

cd cgal_alpha_bindings
mkdir build && cd build
cmake ..
make cgal_alpha_algo2 -j$(sysctl -n hw.ncpu)
```

**Conda**

```bash
conda install -c conda-forge cgal-cpp pybind11

cd cgal_alpha_bindings
mkdir build && cd build
cmake ..
make cgal_alpha_algo2 -j8
```

#### Making the bindings available

After building, the compiled `.so` file lands in `cgal_alpha_bindings/build/`. When you import `cgal_alpha_algo2` in Python, it needs to find that `.so` on `sys.path`. The code does this automatically by looking for `cgal_alpha_bindings/build/` relative to the source tree.

This works out of the box when running from the repo. However, some environments override the working directory or `sys.path` — for example SLURM jobs with `multiprocessing` workers using the `spawn` or `forkserver` start method. In that case each worker process starts fresh and may not inherit the path setup. To handle this, set the environment variable before launching your job:

```bash
export CGAL_BINDINGS_DIR=/path/to/cgal_alpha_bindings/build
export PYTHONPATH="$CGAL_BINDINGS_DIR:$PYTHONPATH"
```

### Curvature extension

The `cpp_curvature` module computes principal curvatures on alpha complex surfaces. It is exactly the same as `igl.per_vertex_normals` but takes custom vertex normals as input (computed by the alpha complex pipeline). It is required for using alpha complex surfaces.

```bash
cd cpp_curvature
python build.py
```

The `eigen` headers are already available from the `cgal-cpp` conda install, and `pybind11` was installed earlier.

## Datasets

Every task reads its data from a directory passed as `data_dir=`; there is no
default. The commands below fetch the public sources into `data/<name>/`, but any
path works.

**PINDER-Pair.** `preprocess.py` pulls the systems through the `pinder` package
and writes the PDBs and split CSVs:

```bash
python alphasurf/tasks/pinder_pair/preprocess.py \
  --output_dir data/pinder-pair \
  --test_setting all \
  --num_workers 30
```

**MISATO** (`MD.hdf5` is ~133 GB):

```bash
mkdir -p data/misato/splits
wget -c -O data/misato/MD.hdf5 https://zenodo.org/records/7711953/files/MD.hdf5
for split in train val test; do
  wget -O data/misato/splits/${split}.txt \
    https://zenodo.org/records/7711953/files/${split}_MD.txt
done
```

**CATH** (used for S3F pretraining):

```bash
mkdir -p data/cath && cd data/cath
curl -fL -o dompdb.tar https://huggingface.co/datasets/tyang816/cath/resolve/main/dompdb.tar
tar -xf dompdb.tar
```

**ProteinGym** v1.3 substitutions and the matching AF2 structures:

```bash
mkdir -p data/proteingym && cd data/proteingym
BASE=https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3
curl -fL -o DMS_ProteinGym_substitutions.zip $BASE/DMS_ProteinGym_substitutions.zip
curl -fL -o ProteinGym_AF2_structures.zip    $BASE/ProteinGym_AF2_structures.zip
unzip -q DMS_ProteinGym_substitutions.zip -d substitutions
unzip -q ProteinGym_AF2_structures.zip    -d af2_structures
curl -fL -o substitutions/DMS_substitutions.csv \
  https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv
```

**MaSIF-Ligand.** Obtain the raw release from
[MaSIF](https://github.com/LPDI-EPFL/masif) and arrange it as
`<data_dir>/raw_data_MasifLigand/{pdb,ligand,splits}/`; preprocessed surfaces are
written to `<data_dir>/dataset_MasifLigand/`.

## Tasks

### MasifLigand

Prediction of ligand binding sites on protein surfaces. Given a protein structure, the model classifies surface patches by ligand type (7 classes).

**Location:** `alphasurf/tasks/masif_ligand_new/`

Supports both on-the-fly and disk-based training. On-the-fly mode generates surfaces and graphs during training, allowing experimentation with different surface methods without re-preprocessing.

```bash
cd alphasurf/tasks/masif_ligand_new

# On-the-fly training with alpha complex surfaces
python train.py \
  data_dir=/path/to/masif_ligand \
  on_fly.surface_method=alpha_complex \
  on_fly.alpha_value=0 \
  on_fly.face_reduction_rate=1.0
```

### PINDER-Pair

Protein-protein interaction prediction on the [PINDER](https://pinder.org/) dataset. Given a receptor and ligand protein, the model predicts per-residue interaction probabilities (which residue pairs form the interface) and per-residue binding site scores.

**Location:** `alphasurf/tasks/pinder_pair/`

Supports both on-the-fly and disk-based training. On-the-fly mode generates surfaces and graphs during training. Three test settings are available: holo (bound structures), apo (unbound experimental), and af2 (AlphaFold2 predicted).

```bash
cd alphasurf/tasks/pinder_pair

# On-the-fly training
python train.py \
  data_dir=/path/to/pinder \
  on_fly.surface_method=alpha_complex \
  on_fly.face_reduction_rate=1.0 \
  on_fly.use_whole_surfaces=True \
  cfg_surface.use_whole_surfaces=True \
  cfg_graph.use_graphs=True \
  cfg_graph.use_esm=False \
  encoder=pronet_gvpencoder.yaml \
  optimizer.lr=0.0001 \
  scheduler=reduce_lr_on_plateau \
  epochs=500 \
  loader.batch_size=4 \
  loader.num_workers=8 \
  loader.pin_memory=false \
  loader.persistent_workers=true

# Disk-based training (requires precompute.py first)
python precompute.py data_dir=/path/to/pinder
python train.py data_dir=/path/to/pinder on_fly=null
```

#### Noise augmentation

PINDER-Pair supports `joint_mesh` train-time augmentation when surfaces are generated on the fly. It first adds Gaussian noise to the atom coordinates used by both the residue graph and surface generator, then independently displaces the resulting surface vertices along their normals. Validation and testing always use clean structures. Set `noise_mode=none` to disable augmentation.

Train with `joint_mesh` noise:

```bash
cd alphasurf/tasks/pinder_pair

python train.py \
  data_dir=/path/to/pinder \
  on_fly.surface_method=alpha_complex \
  on_fly.noise_mode=joint_mesh \
  on_fly.sigma_graph=0.3 \
  on_fly.sigma_mesh=0.3 \
  on_fly.clip_sigma=3.0
```

A trained checkpoint can be evaluated on all three clean structural settings with:

```bash
python test.py \
  data_dir=/path/to/pinder \
  ckpt_path=/path/to/model.ckpt \
  test_setting=all
```

### MISATO binding-site prediction

Residue-level ligand binding-site prediction on the [MISATO](https://zenodo.org/records/7711953) molecular-dynamics dataset. The ligand is used only to construct fixed binary labels: a residue is positive when its C-alpha atom is within 10 Å of a ligand heavy atom in trajectory frame 0. The model receives only the protein graph and alpha-complex surface.

The official sequence-clustered train, validation, and test splits are applied at the complex level. Training samples a random trajectory frame, while validation and testing use frame 0.

**Location:** `alphasurf/tasks/misato_binding_site/`

After downloading the trajectory file and official splits (see
[Datasets](#datasets)), preprocessing reads frame 0 and writes
`binding_site/<pdb_id>.pt` files containing the protein atom metadata, residue
indices, and fixed binding-site labels required for training:

```bash
python -m alphasurf.tasks.misato_binding_site.preprocess \
  --data-dir /path/to/misato
```

The trajectory coordinates remain in `MD.hdf5`; training reads one frame per complex lazily instead of copying trajectories into the preprocessed cache.

Train with random MD frames and evaluate on frame 0:

```bash
python -m alphasurf.tasks.misato_binding_site.train \
  data_dir=/path/to/misato \
  train_frame_mode=random \
  eval_frame_mode=first
```

Publication evaluation uses Guo et al.'s factorized batch-64 aggregation:
systems remain in test-split order, residue predictions are pooled within each
64-system chunk, and chunk metrics are averaged with residue-count weights.
The implementation is in
[`evaluate_guo_batch64.py`](alphasurf/tasks/misato_binding_site/evaluate_guo_batch64.py).

See the [MISATO task README](alphasurf/tasks/misato_binding_site/README.md) for additional evaluation utilities.

### MISATO affinity prediction

Binding-affinity regression on the same MISATO complexes.

**Location:** `alphasurf/tasks/misato_affinity/`

```bash
python -m alphasurf.tasks.misato_affinity.build_affinity \
  --csv /path/to/misato/affinity_data.csv --out /path/to/misato/affinity.h5
python -m alphasurf.tasks.misato_affinity.preprocess --data-dir /path/to/misato
python -m alphasurf.tasks.misato_affinity.train data_dir=/path/to/misato
```

### S3F pretraining on CATH

Self-supervised structure-and-surface pretraining on CATH domains, following S3F.
It produces the checkpoints scored by the ProteinGym task.

**Location:** `alphasurf/tasks/s3f_pretrain/`

Surfaces follow the repo-wide `on_fly` convention: leave `on_fly` set to generate
them at runtime, or set `on_fly=null` to read precomputed point clouds from
`precompute_dir`.

```bash
cd alphasurf/tasks/s3f_pretrain

python train.py data_dir=/path/to/cath/dompdb
```

For the precomputed path, build the clouds first and point `precompute_dir` at them:

```bash
python precompute_s3f_exact.py \
  --pdb_dir /path/to/cath/dompdb \
  --output_dir /path/to/cath/s3f_exact_precomputed

python train.py \
  data_dir=/path/to/cath/dompdb \
  precompute_dir=/path/to/cath/s3f_exact_precomputed \
  on_fly=null
```

`precompute_alpha.py` is the alpha-complex equivalent; it writes
`<parent of data_dir>/{surfaces,graphs}/<method>_<face_reduction_rate>_a<alpha>/`.

### ProteinGym

Zero-shot fitness prediction on the 217 ProteinGym substitution assays, scoring
masked mutant-versus-wild-type log-odds with an S3F-pretrained checkpoint.

**Location:** `alphasurf/tasks/proteingym/`

```bash
cd alphasurf/tasks/proteingym

python evaluate.py \
  --ckpt /path/to/s3f_pretrain.ckpt \
  --substitutions-dir /path/to/proteingym/substitutions \
  --af2-dir /path/to/proteingym/af2_structures \
  --output-dir runs/alphasurf
```

`summary.csv` records, per assay, how much of it was scored structurally
(`num_scored`, `num_groups_geometry_failed`, `num_positions_low_plddt`); read
those next to the Spearman correlation. See the
[ProteinGym task README](alphasurf/tasks/proteingym/README.md).

## Inference

Embed a trained model's encoder on a single protein to get per-residue graph
embeddings and per-vertex surface embeddings.

**Location:** `alphasurf/tasks/inference/`

```bash
cd alphasurf/tasks/inference

python embed.py --ckpt /path/to/model.ckpt --pdb protein.pdb
```

The checkpoint is a `PinderPairModule` checkpoint, produced by the
[PINDER-Pair](#pinder-pair) task. No weights are distributed with this repository.

Output is a `.pt` file containing `graph_embedding` (N_residues x D),
`surface_embedding` (N_verts x D), `graph_node_pos`, and `surface_verts`.

## Reproducing the figures

Figure scripts live in `plotting/`, grouped by subject, and write to
`plotting/figures/<group>/`. Each script resolves its inputs from the repo root,
so it can be run from any working directory.

```bash
python plotting/pinder_pair/plot_perf_vs_throughput_seeds.py
python plotting/masif_ligand/plot_perf_vs_throughput_combined.py
```

The spectral figures come from `scripts/`, which computes before it draws. The
first command is the expensive one; the other three read its output:

```bash
python scripts/spectral_comparison.py \
  --pdb-dir /path/to/pinder/pdb \
  --output-dir scripts/outputs/spectral_heat \
  --workers 30

python scripts/plot_spectral_distributions.py \
  --input-dir scripts/outputs/spectral_heat --output-dir scripts/outputs/spectral_heat
python scripts/plot_kernel_profile.py  --input-dir scripts/outputs/spectral_heat
python scripts/plot_dirac_diffusion.py --pdb /path/to/protein.pdb
```

`plot_dirac_diffusion.py` draws its panels with PyMOL by default; pass
`--render mesh` or `--render vector` to draw them with matplotlib instead. Its
`sas` and `sas_dec` mesh kinds need the `cgal_sbl_sampling` module, which
requires `SBL_ROOT` at build time; every other kind works without it.

The MaSIF-Ligand combined panels read the PINDER summary CSVs from
`plotting/figures/pinder_pair/`, so run the PINDER scripts first. Every PINDER
AUROC figure is computed on the frozen common system set defined in
`plotting/pinder_pair/common_systems.py` (1835 holo, 309 apo, 1582 af2); a run
that does not cover that set is rejected. See
[`plotting/README.md`](plotting/README.md).
