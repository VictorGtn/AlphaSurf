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

# Augmentation settings
N_AUGMENTED_VIEWS=20          # Number of augmented views per surface (1 = no augmentation)
AUGMENTATION_SIGMA=0.15        # Noise standard deviation in Angstroms
AUGMENTATION_NOISE_TYPE="isotropic"  # 'normal' or 'isotropic'

# Surface generation settings
SURFACE_METHOD="alpha_complex"         # 'msms' or 'alpha_complex'
FACE_REDUCTION_RATE=1.0       # Mesh simplification rate (1.0 = no simplification)

# --- Script Execution ---
echo "=============================================="
echo "Starting surface preprocessing with augmentation"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Surface method: $SURFACE_METHOD"
echo "  Face reduction rate: $FACE_REDUCTION_RATE"
echo ""
echo "Augmentation settings:"
echo "  Number of views: $N_AUGMENTED_VIEWS"
echo "  Sigma: $AUGMENTATION_SIGMA Å"
echo "  Noise type: $AUGMENTATION_NOISE_TYPE"
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
    preprocessing.face_reduction_rate=$FACE_REDUCTION_RATE \
    cfg_surface.n_augmented_views=$N_AUGMENTED_VIEWS \
    cfg_surface.augmentation_sigma=$AUGMENTATION_SIGMA \
    cfg_surface.augmentation_noise_type=$AUGMENTATION_NOISE_TYPE"

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
