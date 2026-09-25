# AlphaSurf: On-the-Fly Surface Computations for Protein Representation Learning

Reference implementation.

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
    - [S3F pretraining on CATH](#s3f-pretraining-on-cath)
    - [ProteinGym](#proteingym)

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

Every task takes its data directory as `data_dir=`; there is no default.

**PINDER-Pair** — PDBs and split CSVs via the `pinder` package:

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

**CATH** (S3F pretraining):

```bash
mkdir -p data/cath && cd data/cath
curl -fL -o dompdb.tar https://huggingface.co/datasets/tyang816/cath/resolve/main/dompdb.tar
tar -xf dompdb.tar
```

**ProteinGym** v1.3 substitutions plus AF2 structures:

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

**MaSIF-Ligand** — take the raw release from
[MaSIF](https://github.com/LPDI-EPFL/masif) and lay it out as
`<data_dir>/raw_data_MasifLigand/{pdb,ligand,splits}/`. Preprocessed surfaces go
to `<data_dir>/dataset_MasifLigand/`.

## Tasks

### MasifLigand

Classifies surface patches by ligand type (7 classes).

**Location:** `alphasurf/tasks/masif_ligand_new/`

On-the-fly mode generates surfaces and graphs during training, so changing surface
method needs no re-preprocessing.

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

Protein-protein interaction on [PINDER](https://pinder.org/): per-residue-pair
interaction probabilities and per-residue binding-site scores.

**Location:** `alphasurf/tasks/pinder_pair/`

Three test settings: holo (bound), apo (unbound experimental), af2 (predicted).

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

`joint_mesh` adds Gaussian noise to the atom coordinates feeding both the graph
and the surface, then displaces the resulting vertices along their normals.
Validation and test always use clean structures; `noise_mode=none` disables it.

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

Evaluate a checkpoint on all three clean settings:

```bash
python test.py \
  data_dir=/path/to/pinder \
  ckpt_path=/path/to/model.ckpt \
  test_setting=all
```

### MISATO binding-site prediction

Residue-level binding-site prediction on [MISATO](https://zenodo.org/records/7711953).
A residue is positive when its C-alpha lies within 10 Å of a ligand heavy atom at
frame 0; the model itself never sees the ligand. Official sequence-clustered
splits, applied per complex. Training samples a random frame, evaluation uses
frame 0.

**Location:** `alphasurf/tasks/misato_binding_site/`

Preprocessing writes `binding_site/<pdb_id>.pt` with atom metadata, residue
indices and labels:

```bash
python -m alphasurf.tasks.misato_binding_site.preprocess \
  --data-dir /path/to/misato
```

Coordinates stay in `MD.hdf5`; training reads one frame per complex lazily.

```bash
python -m alphasurf.tasks.misato_binding_site.train \
  data_dir=/path/to/misato \
  train_frame_mode=random \
  eval_frame_mode=first
```

Published numbers use Guo et al.'s factorized batch-64 aggregation — systems in
test-split order, residue predictions pooled per 64-system chunk, chunks averaged
by residue count — in
[`evaluate_guo_batch64.py`](alphasurf/tasks/misato_binding_site/evaluate_guo_batch64.py).
See the [task README](alphasurf/tasks/misato_binding_site/README.md).

### S3F pretraining on CATH

Self-supervised pretraining on CATH domains, following S3F. Produces the
checkpoints scored by ProteinGym.

**Location:** `alphasurf/tasks/s3f_pretrain/`

Leave `on_fly` set to build surfaces at runtime, or `on_fly=null` to read
precomputed clouds from `precompute_dir`.

```bash
cd alphasurf/tasks/s3f_pretrain

python train.py data_dir=/path/to/cath/dompdb
```

For the precomputed path, build the clouds first:

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

Zero-shot fitness on the 217 substitution assays: masked mutant-versus-wild-type
log-odds from an S3F-pretrained checkpoint.

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
