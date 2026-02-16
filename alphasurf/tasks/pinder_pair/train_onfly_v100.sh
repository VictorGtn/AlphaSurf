#!/bin/bash
#SBATCH --job-name=atomsurf_onfly
#SBATCH --output=/lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/log/train_onfly_%j.log
#SBATCH --error=/lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/log/train_onfly_%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --account=pyg@v100
#SBATCH --constraint=v100-32g

# 1. Load Environment
module purge
module load anaconda-py3

# 2. Activate Conda
eval "$(conda shell.bash hook)"
conda activate /lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf_env

# 3. Exports
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export WANDB_MODE=offline
export WANDB_DIR=$(pwd)/wandb_logs
mkdir -p $WANDB_DIR

export REPO_ROOT=/lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/atomsurf
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$(dirname $REPO_ROOT)/cgal_alpha_bindings/build

# Logic to switch between Disk (MSMS) and OnFly (Alpha)
if [ "$MODE" == "disk" ]; then
    ON_FLY_ARGS="on_fly=null cfg_surface.data_name=surfaces_msms_fr0.1"
else
    ON_FLY_ARGS="on_fly.surface_method=${SURFACE_METHOD:-alpha_complex} \
                 on_fly.alpha_value=${ALPHA_VALUE:-0} \
                 on_fly.noise_mode=${NOISE_MODE:-none} \
                 on_fly.sigma_graph=${SIGMA_GRAPH:-0.3} \
                 on_fly.sigma_mesh=${SIGMA_MESH:-0.3} \
                 on_fly.clip_sigma=${CLIP_SIGMA:-null} \
                 on_fly.min_vert_number=16"
fi

# Run training
python $REPO_ROOT/atomsurf/tasks/pinder_pair/train.py \
    run_name=${RUN_NAME:-exp_v100} \
    use_wandb=true \
    data_dir=/lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/atomsurf/data/pinder-pair \
    hydra.searchpath=[file:///lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/atomsurf/atomsurf/tasks/shared_conf] \
    $ON_FLY_ARGS \
    encoder=pronet_gvpencoder \
    optimizer.lr=0.0005 \
    epochs=${EPOCHS:-30} \
    interface_distance_graph=5.0 \
    interface_distance_surface=${INTERFACE_DIST_SURF:-3.8} \
    loader.num_workers=${NUM_WORKERS:-8} \
    loader.use_dynamic_batching=true \
    loader.max_atoms_per_batch=${MAX_ATOMS:-20000} \
    loader.persistent_workers=${PERSISTENT_WORKERS:-true} \
    surface_neg_to_pos_ratio=10.0 \
    train.save_top_k=5
