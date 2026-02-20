#!/bin/bash
#SBATCH --job-name=pinder_precompute
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mem=64000

# Load environment
source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf

# Set path
export REPO_ROOT=$(git rev-parse --show-toplevel)
# Ensure cgal_alpha bindings are in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$(dirname $REPO_ROOT)/cgal_alpha_bindings/build

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
param_surface="alpha_complex"
param_alpha=0.0
param_reduction=1.0
# param_surface="msms"

echo "Surface Method: $param_surface"
echo "Alpha Value:    $param_alpha"
echo "Face Reduction: $param_reduction"
echo ""

python alphasurf/tasks/pinder_pair/precompute.py \
    on_fly.surface_method=$param_surface \
    on_fly.alpha_value=$param_alpha \
    on_fly.face_reduction_rate=$param_reduction \
    preprocessing.recompute_surfaces=true \
    loader.num_workers=20

echo ""
echo "============================================================"
echo "PRECOMPUTATION COMPLETE!"
echo "============================================================"
