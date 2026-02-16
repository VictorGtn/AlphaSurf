#!/bin/bash
#SBATCH --job-name=pinder_preprocess
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mem=64000

# Load environment
source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf

# Set path
export REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$(dirname $REPO_ROOT)/cgal_alpha_bindings/build

# Limit threads to match worker count
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Navigate to task directory
cd $REPO_ROOT/atomsurf/tasks/pinder_pair

# Preprocess MSMS surfaces + graphs for all splits (train, val, test)

echo "============================================================"
echo "PREPROCESSING PINDER-PAIR: MSMS SURFACES + GRAPHS"
echo "============================================================"

# Configuration
FACE_REDUCTION=0.1  # Set to lower value (e.g., 0.5) for coarser meshes
MAX_SYSTEMS=null    # Set to small number (e.g., 10) for testing, null for all

# Training split
echo ""
echo "Processing TRAINING split..."
python preprocess.py \
  preprocessing.surface_method=msms \
  preprocessing.face_reduction_rate=$FACE_REDUCTION \
  preprocessing.max_systems=$MAX_SYSTEMS \
  preprocessing.compute_surfaces=true \
  preprocessing.compute_graphs=true \
  preprocessing.compute_esm=false \
  preprocessing.split=train \
  loader.num_workers=20

# Validation split
echo ""
echo "Processing VALIDATION split..."
python preprocess.py \
  preprocessing.surface_method=msms \
  preprocessing.face_reduction_rate=$FACE_REDUCTION \
  preprocessing.max_systems=$MAX_SYSTEMS \
  preprocessing.compute_surfaces=true \
  preprocessing.compute_graphs=true \
  preprocessing.compute_esm=false \
  preprocessing.split=val \
  loader.num_workers=18

# Test split (holo)
echo ""
echo "Processing TEST split (holo structures)..."
python preprocess.py \
  preprocessing.surface_method=msms \
  preprocessing.face_reduction_rate=$FACE_REDUCTION \
  preprocessing.max_systems=$MAX_SYSTEMS \
  preprocessing.compute_surfaces=true \
  preprocessing.compute_graphs=true \
  preprocessing.compute_esm=false \
  preprocessing.split=test \
  loader.num_workers=18

echo ""
echo "============================================================"
echo "PREPROCESSING COMPLETE!"
echo "============================================================"
echo ""
echo "Generated directories:"
echo "  - surfaces_msms_fr1.0/  (MSMS surfaces)"
echo "  - rgraph/               (Residue graphs)"
echo ""
echo "To use in training, set: on_fly=null"
