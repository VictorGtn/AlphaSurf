#!/bin/bash
#SBATCH --job-name=preprocess_patches
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
# Data subdirectory (required, e.g., msms_01, alpha0, alpha1, alpha10, alpha25)
DATA_SUBDIR="alpha0"

# Number of parallel workers for preprocessing
NUM_WORKERS=8

# Radius for point selection (Angstroms, adaptive)
RADIUS=6.0

# Minimum vertices needed for eigendecomposition
MIN_VERTICES=130

# --- Script Execution ---
echo "Starting patch preprocessing..."
echo "Data subdirectory: $DATA_SUBDIR"
echo "Number of workers: $NUM_WORKERS"
echo "Radius: $RADIUS Å"
echo "Min vertices: $MIN_VERTICES"
echo ""

# Build the python command
CMD="python3 $SLURM_SUBMIT_DIR/preprocess_patches.py --data_subdir $DATA_SUBDIR --num_workers $NUM_WORKERS --radius $RADIUS --min_vertices $MIN_VERTICES"

# Run the preprocessing script
echo "Running command: $CMD"
$CMD

echo "Preprocessing script finished."



