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
#SBATCH --output=log/pinder_test/%x_%j.out
#SBATCH --error=log/pinder_test/%x_%j.err
#SBATCH --hint=nomultithread

# submission command:
# CKPT_PATH=/path/to/ckpt.ckpt TEST_SETTING=af2 sbatch test.sh

# 1. Load H100 Environment
module purge
module load arch/h100
module load anaconda-py3

export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/pyg/ust26qt}

# 2. Activate Conda
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/atomsurf_h100_new

# 3. Temp & cache to SCRATCH
export TMPDIR=$SCRATCH/tmp && mkdir -p $TMPDIR
export PYKEOPS_CACHE_DIR=$SCRATCH/keops_cache && mkdir -p $PYKEOPS_CACHE_DIR

# 4. Exports
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export WANDB_MODE=offline

export REPO_ROOT=/lustre/fsn1/projects/rech/pyg/ust26qt/alphasurf/alphasurf/alphasurf
export CGAL_BINDINGS_DIR=$REPO_ROOT/../cgal_alpha_bindings/build
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT/..:$CGAL_BINDINGS_DIR

# 5. Test CGAL
python -c "import sys; sys.path.insert(0, '$CGAL_BINDINGS_DIR'); import cgal_alpha_algo2; print('[CGAL OK]', cgal_alpha_algo2.__file__)"

# 6. On-the-fly args (no noise, no igl normals)
ON_FLY_ARGS="on_fly.surface_method=${SURFACE_METHOD:-alpha_complex} \
             on_fly.alpha_value=${ALPHA_VALUE:-0} \
             on_fly.noise_mode=none \
             on_fly.min_vert_number=16 \
             on_fly.use_igl_normals=false \
             on_fly.face_reduction_rate=${FACE_REDUCTION_RATE:-1.0} \
             on_fly.nanoshaper_grid_scale=${NANOSHAPER_GRID_SCALE:-0.3} \
             +on_fly.edtsurf_grid_scale=${EDTSURF_GRID_SCALE:-0.5} \
             on_fly.max_vert_number=${MAX_VERT_NUMBER:-100000}"

# 7. Run testing
python -u -X faulthandler $REPO_ROOT/tasks/pinder_pair/test.py \
    test_setting=${TEST_SETTING:-af2} \
    data_dir=$REPO_ROOT/../data/pinder-pair \
    hydra.searchpath=[file://$REPO_ROOT/tasks/shared_conf] \
    $ON_FLY_ARGS \
    encoder=pronet_gvpencoder \
    interface_distance_graph=5.0 \
    interface_distance_surface=${INTERFACE_DIST_SURF:-3.8} \
    loader.num_workers=${NUM_WORKERS:-16} \
    surface_neg_to_pos_ratio=10.0 \
    "+ckpt_path='${CKPT_PATH}'"
