#!/bin/bash -l
#SBATCH --job-name=atomsurf_onfly
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/train_onfly_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/train_onfly_%j.err
#SBATCH --time=2-00:00:00        
#SBATCH --mem=32000              
#SBATCH --gres=gpu:1             
#SBATCH -p cbio-gpu              
#SBATCH --cpus-per-task=2      # 8 workers + 1 main process
#SBATCH --nodelist=node006

source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Add cgal_alpha_bindings to PYTHONPATH (needed for multiprocessing workers)
export PYTHONPATH="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/cgal_alpha_bindings/build:$PYTHONPATH"

# Configuration for WandB Offline Mode
export WANDB_MODE=offline
export WANDB_DIR="/cluster/CBIO/data2/vgertner/atomsurf/log"
mkdir -p $WANDB_DIR

# Limit threads per worker to avoid oversubscription with multiprocessing
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Change to script directory to ensure relative paths work
cd /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/atomsurf/tasks/masif_ligand

# =============================================================================
# ENVIRONMENT OVERRIDES (For Batch Launching)
# =============================================================================
SURFACE_METHOD="${SURFACE_METHOD:-msms}"
ALPHA_VALUE="${ALPHA_VALUE:-0.0}"
FACE_REDUCTION_RATE="${FACE_REDUCTION_RATE:-1.0}"
USE_WHOLE_SURFACES="${USE_WHOLE_SURFACES:-False}"
PATCH_RADIUS="${PATCH_RADIUS:-8.0}"
NOISE_MODE="${NOISE_MODE:-independent}"
SIGMA_GRAPH="${SIGMA_GRAPH:-0.3}"
SIGMA_MESH="${SIGMA_MESH:-0.0}"
CLIP_SIGMA="${CLIP_SIGMA:-null}"
DEBUG_SAVE_PLY="${DEBUG_SAVE_PLY:-true}"
# DEBUG_PLY_DIR="/lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/atomsurf/data/masif_ligand/onfly_debug"
DEBUG_PLY_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/onfly_debug"
DEBUG_MAX_SAMPLES=-1
DEBUG_EXIT_AFTER_SAVE="${DEBUG_EXIT_AFTER_SAVE:-false}"
precomputed_patches_dir="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/msms_01/surfaces_patches_geom"

# Data directory
# DATA_DIR="/lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/atomsurf/data/masif_ligand"
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand"
ESM_DIR="${DATA_DIR}/esm"

# Run name
RUN_NAME="${RUN_NAME:-onfly_${SURFACE_METHOD}_alpha${ALPHA_VALUE}_noise${NOISE_MODE}_sigma${SIGMA_GRAPH}_${SIGMA_MESH}}"

echo "========================================="
echo "CLEANING SHM FOR USER: ust26qt"
find /dev/shm -user ust26qt -delete 2>/dev/null || true
echo "========================================="

echo "=============================================="
echo "ON-THE-FLY TRAINING"
echo "=============================================="
echo "Run name: $RUN_NAME"
echo "Surface method: $SURFACE_METHOD"
echo "Alpha value: $ALPHA_VALUE"
echo "Face reduction rate: $FACE_REDUCTION_RATE"
echo "Noise mode: $NOISE_MODE"
echo "=============================================="

python3 train_on_fly.py \
  data_dir=$DATA_DIR \
  cfg_surface.use_whole_surfaces=$USE_WHOLE_SURFACES \
  cfg_graph.use_graphs=True \
  cfg_graph.use_esm=True \
  on_fly.esm_dir=$ESM_DIR \
  on_fly.precomputed_patches_dir=$precomputed_patches_dir \
  on_fly.surface_method=$SURFACE_METHOD \
  on_fly.alpha_value=$ALPHA_VALUE \
  on_fly.face_reduction_rate=$FACE_REDUCTION_RATE \
  on_fly.use_whole_surfaces=$USE_WHOLE_SURFACES \
  on_fly.patch_radius=$PATCH_RADIUS \
  on_fly.noise_mode=$NOISE_MODE \
  on_fly.sigma_graph=$SIGMA_GRAPH \
  on_fly.sigma_mesh=$SIGMA_MESH \
  on_fly.clip_sigma=$CLIP_SIGMA \
  on_fly.debug_save_ply=$DEBUG_SAVE_PLY \
  on_fly.debug_ply_dir=$DEBUG_PLY_DIR \
  on_fly.debug_max_samples=$DEBUG_MAX_SAMPLES \
  on_fly.debug_exit_after_save=$DEBUG_EXIT_AFTER_SAVE \
  encoder=pronet_gvpencoder.yaml \
  optimizer.lr=0.0001 \
  scheduler=reduce_lr_on_plateau \
  epochs=2 \
  loader.batch_size=2 \
  loader.num_workers=8 \
  loader.pin_memory=false \
  loader.persistent_workers=true \
  loader.prefetch_factor=1 \
  diffusion_net.use_bn=true \
  diffusion_net.use_layernorm=false \
  diffusion_net.init_time=2.0 \
  diffusion_net.init_std=2.0 \
  train.save_top_k=5 \
  train.early_stoping_patience=500 \
  run_name=$RUN_NAME \
  +train.profile=false \
  device=0 \
  seed=${SEED:-2024}
