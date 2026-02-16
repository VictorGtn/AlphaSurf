#!/bin/bash
#SBATCH --job-name=preprocess_dmasif
#SBATCH --partition=cbio-gpu
#SBATCH --nodelist=node006
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.err
#SBATCH --gres=gpu:1

# Activate conda environment
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Configure KeOps for GPU compatibility
# Set a job-specific cache directory to avoid conflicts
export PYKEOPS_CACHE_DIR="/tmp/pykeops_cache_${SLURM_JOB_ID}"
mkdir -p "$PYKEOPS_CACHE_DIR"

# Force KeOps to recompile CUDA kernels for the current GPU
export KEOPS_FORCE_COMPILE=1
export KEOPS_FORCE_USE_CPU=0

# Detect GPU compute capability and set CUDA compilation flags
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    echo "Detected GPU: $GPU_NAME"
    
    # Try to get compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)
    if [ -n "$COMPUTE_CAP" ]; then
        echo "GPU Compute Capability: $COMPUTE_CAP"
        # Set CUDA architecture flag for KeOps compilation
        ARCH_NUM=$(echo "$COMPUTE_CAP" | tr '.' '')
        export PYKEOPS_CUDA_COMPILATION_FLAGS="-arch=sm_${ARCH_NUM}"
        echo "Setting CUDA architecture: sm_${ARCH_NUM}"
    fi
fi

# Clear any existing KeOps cache
if [ -d "$HOME/.cache/pykeops" ]; then
    echo "Clearing KeOps cache..."
    rm -rf "$HOME/.cache/pykeops"
fi
if [ -d "$CONDA_PREFIX/.cache/pykeops" ]; then
    echo "Clearing KeOps cache in conda environment..."
    rm -rf "$CONDA_PREFIX/.cache/pykeops"
fi

# --- Configuration ---
# Set the base data directory containing the 'pdb/' folder
# You can change this path if your data is located elsewhere.
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/raw_data_MasifLigand"

# Set the output directory for the generated dmasif surfaces.
# If you want to use a custom location, uncomment and set the line below.
# OUT_DIR="/path/to/your/output/directory"

# Number of parallel workers for preprocessing.
NUM_WORKERS=8


# --- Script Execution ---
echo "Starting dMaSIF surface preprocessing..."
echo "Data directory: $DATA_DIR"
if [ -n "$OUT_DIR" ]; then
    echo "Output directory: $OUT_DIR"
fi
echo "Number of workers: $NUM_WORKERS"
echo ""

# Build the python command
# - num_workers: parallel PDB parsing on CPU
# - batch_size: proteins per GPU batch
CMD="python3 $SLURM_SUBMIT_DIR/preprocess_dmasif.py --data_dir $DATA_DIR --num_workers $NUM_WORKERS --batch_size 8"

# Add output directory to command if specified
if [ -n "$OUT_DIR" ]; then
    CMD="$CMD --out_dir $OUT_DIR"
fi

# Add recompute flag if you want to force reprocessing of existing files
CMD="$CMD --recompute"

# Run the preprocessing script
echo "Running command: $CMD"
$CMD

echo "Preprocessing script finished."
