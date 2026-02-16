#!/bin/bash
#SBATCH --job-name=preprocess_surfaces
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.err

# Activate conda environment
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Add cgal_alpha_bindings to PYTHONPATH (needed for multiprocessing workers)
export PYTHONPATH="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/cgal_alpha_bindings/build:$PYTHONPATH"

# Limit threads per worker to avoid oversubscription with multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# --- Configuration ---
# Surface generation method: 'msms' or 'alpha_complex'
SURFACE_METHOD="alpha_complex"

# Alpha value for alpha_complex method (ignored for msms)
ALPHA_VALUE=0

# Face reduction rate (1.0 = no reduction, 0.5 = reduce to 50%)
FACE_REDUCTION_RATE=1.0

# Number of parallel workers for preprocessing
NUM_WORKERS=0

# Data augmentation settings
N_AUGMENTED_VIEWS=1
AUGMENTATION_SIGMA=0.3
AUGMENTATION_NOISE_TYPE="normal"

# --- Script Execution ---
# Clean up old timing files to ensure accurate stats for this run
rm -f /tmp/atomsurf_timing_*.csv 2>/dev/null

echo "Starting surface preprocessing..."
echo "Surface method: $SURFACE_METHOD"
echo "Alpha value: $ALPHA_VALUE"
echo "Face reduction rate: $FACE_REDUCTION_RATE"
echo "Number of workers: $NUM_WORKERS"
echo "Augmented views: $N_AUGMENTED_VIEWS"
echo "Augmentation sigma: $AUGMENTATION_SIGMA"
echo "Augmentation noise type: $AUGMENTATION_NOISE_TYPE"
echo ""

# Build Hydra overrides
HYDRA_OVERRIDES="preprocessing.surface_method=$SURFACE_METHOD"
HYDRA_OVERRIDES="$HYDRA_OVERRIDES preprocessing.alpha_value=$ALPHA_VALUE"
HYDRA_OVERRIDES="$HYDRA_OVERRIDES preprocessing.face_reduction_rate=$FACE_REDUCTION_RATE"
HYDRA_OVERRIDES="$HYDRA_OVERRIDES loader.num_workers=$NUM_WORKERS"
HYDRA_OVERRIDES="$HYDRA_OVERRIDES cfg_surface.n_augmented_views=$N_AUGMENTED_VIEWS"
HYDRA_OVERRIDES="$HYDRA_OVERRIDES cfg_surface.augmentation_sigma=$AUGMENTATION_SIGMA"
HYDRA_OVERRIDES="$HYDRA_OVERRIDES cfg_surface.augmentation_noise_type=$AUGMENTATION_NOISE_TYPE"

# Run the preprocessing script
CMD="python3 $SLURM_SUBMIT_DIR/preprocess.py $HYDRA_OVERRIDES"
echo "Running command: $CMD"
$CMD

echo "Preprocessing finished."
