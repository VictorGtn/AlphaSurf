#!/bin/bash
#SBATCH --job-name=preprocess_patches_stokes
#SBATCH --partition=cbio-gpu
#SBATCH --nodelist=node006
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.err
#SBATCH --gres=gpu:1

# Activate conda environment
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# --- Configuration ---
# Set the base data directory containing the 'stokes_integrals/' folder
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/dmasif"

# Set the patch directory containing .npz patch files
# If not specified, defaults to ../dataset_MasifLigand relative to data_dir
# PATCH_DIR="/path/to/patches"

# Set the stokes directory containing stokes integrals
# If not specified, defaults to data_dir/stokes_integrals
STOKES_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/stokes_integrals"

# Set the output directory for patch stokes
# If not specified, defaults to data_dir/stokes_patches
OUT_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/dmasif/stokes_patches"

# Number of parallel workers for preprocessing
NUM_WORKERS=8

# Radius for point selection (Angstroms, adaptive)
RADIUS=6.0

# Minimum points needed per patch
MIN_POINTS=20

# Use DBSCAN clustering to extract largest component (set to empty string to disable)
USE_CLUSTERING=""  # Empty by default (clustering disabled)
# USE_CLUSTERING="--use_clustering"  # Uncomment to enable clustering
# USE_CLUSTERING="--no_clustering"  # Explicitly disable clustering

# DBSCAN eps parameter for clustering
EPS=3.0

# --- Script Execution ---
echo "Starting Stokes integrals patch preprocessing..."
echo "Data directory: $DATA_DIR"
if [ -n "$PATCH_DIR" ]; then
    echo "Patch directory: $PATCH_DIR"
fi
if [ -n "$STOKES_DIR" ]; then
    echo "Stokes directory: $STOKES_DIR"
fi
if [ -n "$OUT_DIR" ]; then
    echo "Output directory: $OUT_DIR"
fi
echo "Number of workers: $NUM_WORKERS"
echo "Radius: $RADIUS Å"
echo "Min points: $MIN_POINTS"
echo "Clustering: ${USE_CLUSTERING:-disabled}"
if [ -n "$USE_CLUSTERING" ] && [[ "$USE_CLUSTERING" == *"use_clustering"* ]]; then
    echo "DBSCAN eps: $EPS"
fi
echo ""

# Build the python command
CMD="python3 $SLURM_SUBMIT_DIR/preprocess_patches_stokes.py --data_dir $DATA_DIR --num_workers $NUM_WORKERS --radius $RADIUS --min_points $MIN_POINTS --eps $EPS"

# Add optional directories to command if specified
if [ -n "$PATCH_DIR" ]; then
    CMD="$CMD --patch_dir $PATCH_DIR"
fi
if [ -n "$STOKES_DIR" ]; then
    CMD="$CMD --stokes_dir $STOKES_DIR"
fi
if [ -n "$OUT_DIR" ]; then
    CMD="$CMD --out_dir $OUT_DIR"
fi

# Add clustering flag
if [ -n "$USE_CLUSTERING" ]; then
    CMD="$CMD $USE_CLUSTERING"
fi

# Run the preprocessing script
echo "Running command: $CMD"
$CMD

echo "Preprocessing script finished."

