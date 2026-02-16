#!/bin/bash -l
#SBATCH --job-name=atomsurf_patches_dmasif
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/train_patches_dmasif_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/train_patches_dmasif_%j.err
#SBATCH --time=3-00:00:00        
#SBATCH --mem=32000              
#SBATCH --gres=gpu:1             
#SBATCH -p cbio-gpu              
#SBATCH --cpus-per-task=8   
#SBATCH --nodelist=node006

source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# CUDA library setup for PyKeOps
PYTORCH_CUDA_LIB=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null)
export LD_LIBRARY_PATH=$PYTORCH_CUDA_LIB:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Verify CUDA is available
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
    echo "ERROR: PyTorch CUDA not available. Exiting."
    exit 1
fi

cd /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/atomsurf/tasks/masif_ligand

# Configuration
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/dmasif"
SURFACE_DATA_NAME="surfaces_patches_dmasif"
RGRAPH_DATA_NAME="rgraph"

# Optional: Set checkpoint path to resume training
# CKPT_PATH="/path/to/checkpoint.ckpt"
CKPT_PATH="${1:-}"

echo "Data directory: $DATA_DIR"
echo "Surface data: $SURFACE_DATA_NAME"
echo "Graph data: $RGRAPH_DATA_NAME"

# PyKeOps configuration
export PYKEOPS_VERBOSE=1

# Build training arguments
TRAIN_ARGS=(
  "data_dir=$DATA_DIR"
  "cfg_surface.use_whole_surfaces=False"
  "cfg_surface.use_surfaces=True"
  "cfg_surface.data_name=$SURFACE_DATA_NAME"
  "cfg_surface.data_dir=$DATA_DIR"
  "cfg_surface.cache_curvatures=True"
  "cfg_graph.use_graphs=true"
  "cfg_graph.data_name=$RGRAPH_DATA_NAME"
  "cfg_graph.data_dir=$DATA_DIR"
  "encoder=dmasif_pronet_gvpencoder"
  "dmasif_block.dim_in=128"
  "optimizer.lr=0.0001"
  "scheduler=reduce_lr_on_plateau"
  "epochs=300"
  "loader.batch_size=4"
  "loader.num_workers=0"
  "train.save_top_k=5"
  "train.early_stoping_patience=500"
  "run_name=pronet_only"
  "exclude_failed_patches=False"
  "device=0"
  "seed=2025"
)

# Add checkpoint if specified
if [ -n "$CKPT_PATH" ]; then
    echo "Resuming from checkpoint: $CKPT_PATH"
    TRAIN_ARGS+=("+ckpt_path=$CKPT_PATH")
fi

# Run training
echo "Training command: python3 train.py ${TRAIN_ARGS[*]}"
python3 train.py "${TRAIN_ARGS[@]}"
