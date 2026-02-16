#!/bin/bash
#SBATCH --job-name=atomsurf_alpha5_patches
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha5_patches_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha5_patches_%j.err
#SBATCH --time=2-00:00:00        
#SBATCH --mem=32000              
#SBATCH --gres=gpu:1             
#SBATCH -p cbio-gpu              
#SBATCH --cpus-per-task=8   
#SBATCH --nodelist=node006

source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Force single GPU usage - only make GPU 0 visible to PyTorch
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
echo "Number of GPUs visible to PyTorch: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
echo "PyTorch will use device: $(python3 -c 'import torch; print(torch.cuda.current_device() if torch.cuda.is_available() else "CPU")')"
echo "All CUDA devices: $(python3 -c 'import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [])')"

# Change to script directory to ensure relative paths work
cd /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/atomsurf/tasks/masif_ligand

# Data directory configuration
DATA_DIR_NAME="alpha5"
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/$DATA_DIR_NAME"
SURFACE_DATA_NAME="surfaces_patches_geom"
RGRAPH_DATA_NAME="rgraph"

echo "Launching training for $DATA_DIR_NAME with patches"
echo "Data directory: $DATA_DIR"
echo "Surface data: $SURFACE_DATA_NAME"
echo "Graph data: $RGRAPH_DATA_NAME"


# Optional: Set checkpoint path to resume training
CKPT_PATH="${1:-}"

# Build training arguments
TRAIN_ARGS=(
  "data_dir=$DATA_DIR"
  "cfg_surface.use_whole_surfaces=False"
  "cfg_surface.data_name=$SURFACE_DATA_NAME"
  "cfg_surface.data_dir=$DATA_DIR"
  "cfg_graph.data_name=$RGRAPH_DATA_NAME"
  "cfg_graph.data_dir=$DATA_DIR"
  "encoder=pronet_gvpencoder.yaml"
  "optimizer.lr=0.0001"
  "scheduler=reduce_lr_on_plateau"
  "epochs=300"
  "loader.batch_size=4"
  "loader.num_workers=8"
  "diffusion_net.use_bn=true"
  "diffusion_net.use_layernorm=false"
  "diffusion_net.init_time=2.0"
  "diffusion_net.init_std=2.0"
  "train.save_top_k=5"
  "train.early_stoping_patience=500"
  "run_name=hybrid_gvp_3layers_alpha0_patches"
  "+loader.use_dynamic_batching=True"
  "+loader.max_atoms_per_batch=150000"
  "+loader.min_batch_size=2"
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