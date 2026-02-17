#!/bin/bash
#SBATCH --job-name=pinder_train
#SBATCH --partition=cbio-gpu
#SBATCH --nodelist=node006
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mem=32000

# Load environment
source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
# Use only the SLURM-assigned GPU
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-0}}
echo "=== Assigned GPU: $CUDA_VISIBLE_DEVICES (look for gpu.$CUDA_VISIBLE_DEVICES.* in wandb) ==="

# Set path
export REPO_ROOT=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$(dirname $REPO_ROOT)/cgal_alpha_bindings/build

# Configuration for WandB
export WANDB_MODE=online
export WANDB_DIR="/cluster/CBIO/data2/vgertner/atomsurf/log"
export WANDB_CORE="True"
mkdir -p $WANDB_DIR

# Training command
# We use the config defaults which point to the correct data_dir
python $REPO_ROOT/alphasurf/tasks/pinder_pair/train.py \
    run_name=pinder_pair_v1 \
    use_wandb=true \
    loader.num_workers=16 \
    loader.batch_size=16 \
    loader.pin_memory=false \
    loader.persistent_workers=false \
    loader.prefetch_factor=1 \
    loader.use_dynamic_batching=false \
    loader.max_atoms_per_batch=40000 \
    loader.min_batch_size=2 \
    on_fly.surface_method=alpha_complex \
    on_fly.alpha_value=0 \
    on_fly.face_reduction_rate=1.0 \
    on_fly.min_vert_number=16 \
    on_fly.noise_mode=none \
    on_fly.sigma_graph=0.3 \
    on_fly.sigma_mesh=0.0 \
    on_fly.clip_sigma=3.0 \
    diffusion_net.use_bn=true \
    diffusion_net.use_layernorm=false \
    diffusion_net.init_time=2.0 \
    diffusion_net.init_std=2.0 \
    optimizer.lr=0.0001 \
    scheduler=reduce_lr_on_plateau \
    train.save_top_k=5 \
    +train.profile=false

