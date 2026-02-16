#!/bin/bash
#SBATCH --job-name=pinder_test
#SBATCH -A pyg@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --hint=nomultithread

# submission command:
# SBATCH_OPTS="--time=2:00:00" CKPT_PATH=/path/to/ckpt.ckpt TEST_SETTING=af2 sbatch test.sh

# 1. Load H100 Environment
module purge
module load arch/h100
module load anaconda-py3

# 2. Activate Conda
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/atomsurf_h100_env

# 2.5. Redirect temp to SCRATCH
export TMPDIR=$SCRATCH/tmp
mkdir -p $TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR

# KeOps cache
export PYKEOPS_CACHE_DIR=$SCRATCH/keops_cache
mkdir -p $PYKEOPS_CACHE_DIR

# 3. Exports
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export WANDB_MODE=offline

export REPO_ROOT=$SCRATCH/atomsurf_h100/atomsurf
export CGAL_BINDINGS_DIR=$REPO_ROOT/cgal_alpha_bindings/build
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$CGAL_BINDINGS_DIR

# Logic to switch between Disk (MSMS) and OnFly (Alpha)
if [ "$MODE" == "disk" ]; then
    ON_FLY_ARGS="on_fly=null cfg_surface.data_name=surfaces_msms_fr0.1"
else
    ON_FLY_ARGS="on_fly.surface_method=${SURFACE_METHOD:-alpha_complex} \
                 on_fly.alpha_value=${ALPHA_VALUE:-0} \
                 on_fly.noise_mode=none \
                 on_fly.sigma_graph=0.3 \
                 on_fly.sigma_mesh=0.3 \
                 on_fly.clip_sigma=null \
                 on_fly.min_vert_number=16"
fi

# Run testing
python -X faulthandler $REPO_ROOT/atomsurf/tasks/pinder_pair/test.py \
    test_setting=${TEST_SETTING:-af2} \
    data_dir=$SCRATCH/atomsurf_h100/atomsurf/data/pinder-pair \
    'hydra.searchpath=[file://'$SCRATCH'/atomsurf_h100/atomsurf/atomsurf/tasks/shared_conf]' \
    $ON_FLY_ARGS \
    encoder=pronet_gvpencoder \
    interface_distance_graph=5.0 \
    interface_distance_surface=${INTERFACE_DIST_SURF:-3.8} \
    loader.num_workers=${NUM_WORKERS:-16} \
    surface_neg_to_pos_ratio=10.0 \
    "+ckpt_path='${CKPT_PATH}'"
