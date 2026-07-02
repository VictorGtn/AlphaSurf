#!/bin/bash
#SBATCH --job-name=preprocess_surfaces
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.err

# Activate conda environment
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# --- Configuration ---
# Data directory path (relative to the script, or absolute)
DATA_DIR="../../../data/masif_ligand/msms_01"

# Surface generation settings
SURFACE_METHOD="alpha_complex"         # 'msms' or 'alpha_complex'
FACE_REDUCTION_RATE=1.0       # Mesh simplification rate (1.0 = no simplification)

# --- Script Execution ---
echo "=============================================="
echo "Starting surface preprocessing"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Surface method: $SURFACE_METHOD"
echo "  Face reduction rate: $FACE_REDUCTION_RATE"
echo ""
echo "SLURM settings:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo ""

# Change to script directory
cd $SLURM_SUBMIT_DIR

# Build the python command with Hydra overrides
CMD="python3 preprocess.py \
    data_dir=$DATA_DIR \
    preprocessing.surface_method=$SURFACE_METHOD \
    preprocessing.face_reduction_rate=$FACE_REDUCTION_RATE"

# Run the preprocessing script
echo "Running command:"
echo "$CMD"
echo ""

$CMD

EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Preprocessing completed successfully!"
else
    echo "Preprocessing failed with exit code: $EXIT_CODE"
fi
echo "=============================================="

exit $EXIT_CODE
