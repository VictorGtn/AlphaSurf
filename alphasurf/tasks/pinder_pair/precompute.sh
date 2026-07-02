#!/bin/bash
#SBATCH --job-name=pinder_precompute
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=log/pinder_precompute/%x_%j.log
#SBATCH --error=log/pinder_precompute/%x_%j.err
#SBATCH --mem=64000

# Load environment
source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf

# Set path
export REPO_ROOT=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$REPO_ROOT/cgal_alpha_bindings/build_py310

# Limit threads to avoid contention with multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PINDER_DATA_DIR=/cluster/CBIO/data2/vgertner/alphasurf/atomsurf/data/pinder-pair

# Navigate to repo root (precompute.py expects to be run from there or handles imports correctly)
cd $REPO_ROOT

echo "============================================================"
echo "PRECOMPUTING PINDER DATA (Alpha Complex)"
echo "============================================================"

# Configuration
param_surface=${SURFACE_METHOD:-alpha_complex}
param_alpha=0.0
param_reduction=${FACE_REDUCTION:-1.0}
param_nanoshaper_grid_scale=${NANOSHAPER_GRID_SCALE:-0.3}

echo "Surface Method: $param_surface"
echo "Alpha Value:    $param_alpha"
echo "Face Reduction: $param_reduction"
echo "NanoShaper Grid Scale: $param_nanoshaper_grid_scale"
echo ""

python alphasurf/tasks/pinder_pair/precompute.py \
    on_fly.surface_method=$param_surface \
    on_fly.alpha_value=$param_alpha \
    on_fly.face_reduction_rate=$param_reduction \
    on_fly.nanoshaper_grid_scale=$param_nanoshaper_grid_scale \
    on_fly.use_igl_normals=false \
    preprocessing.recompute_surfaces=true \
    loader.num_workers=${SLURM_CPUS_PER_TASK:-40}

echo ""
echo "============================================================"
echo "PRECOMPUTATION COMPLETE!"
echo "============================================================"
