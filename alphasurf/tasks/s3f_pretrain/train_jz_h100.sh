#!/bin/bash
#SBATCH --job-name=s3f_jz
#SBATCH -A pyg@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --requeue
#SBATCH --signal=SIGUSR1@300
#SBATCH --open-mode=append
#SBATCH --hint=nomultithread
#SBATCH --output=/dev/null
#SBATCH --error=/lustre/fsn1/projects/rech/pyg/ust26qt/alphasurf/alphasurf/alphasurf/tasks/s3f_pretrain/log/s3f_jz/%x_%j.err

set -euo pipefail

module purge
module load arch/h100
module load anaconda-py3

export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/pyg/ust26qt}
export WORK=${WORK:-/lustre/fswork/projects/rech/pyg/ust26qt}
export REPO_ROOT=$SCRATCH/alphasurf/alphasurf/alphasurf
export CGAL_BINDINGS_DIR=$REPO_ROOT/../cgal_alpha_bindings/build

eval "$(conda shell.bash hook)"
conda activate "$WORK/atomsurf_h100_new2"

ulimit -n 65536

export TMPDIR=$SCRATCH/tmp
export PYKEOPS_CACHE_DIR=$SCRATCH/keops_cache
mkdir -p "$TMPDIR" "$PYKEOPS_CACHE_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT/..:$CGAL_BINDINGS_DIR"

LOG_DIR=$REPO_ROOT/tasks/s3f_pretrain/log
mkdir -p "$LOG_DIR" "$LOG_DIR/s3f_jz"

RUN_NAME=${RUN_NAME:-s3f_jz}
exec >>"$LOG_DIR/${RUN_NAME}_${SLURM_JOB_ID}.out"

# Keep checkpoint, TensorBoard, and W&B identities stable across requeues.
# Reuse EXPERIMENT_ID explicitly when continuing via a newly submitted job.
export EXPERIMENT_ID=${EXPERIMENT_ID:-${SLURM_JOB_ID:-$(date +%s)}}
export ATOMSURF_VERSION=$EXPERIMENT_ID
export ATOMSURF_WANDB_ID=$EXPERIMENT_ID
export ATOMSURF_RESUME=True

python -c "import sys; sys.path.insert(0, '$CGAL_BINDINGS_DIR'); import cgal_alpha_algo2; print('[CGAL OK]', cgal_alpha_algo2.__file__)"

CKPT_ARGS=()
if [[ -n "${CKPT:-}" ]]; then
    CKPT_ARGS+=("ckpt_path=$CKPT")
fi

cd "$REPO_ROOT"

srun --export=ALL "$CONDA_PREFIX/bin/python" -m alphasurf.tasks.s3f_pretrain.train \
    device=0 \
    run_name="$RUN_NAME" \
    log_dir="$LOG_DIR" \
    data_dir="$REPO_ROOT/../data/cath/dompdb" \
    hydra.searchpath="[file://$REPO_ROOT/tasks/shared_conf]" \
    on_fly.surface_method="${SURFACE_METHOD:-alpha_complex}" \
    on_fly.alpha_value="${ALPHA_VALUE:-0}" \
    on_fly.tufting="${TUFTING:-true}" \
    epochs="${EPOCHS:-100}" \
    loader.num_workers="${NUM_WORKERS:-20}" \
    loader.batch_size="${BATCH_SIZE:-8}" \
    loader.persistent_workers="${PERSISTENT_WORKERS:-true}" \
    loader.prefetch_factor="${PREFETCH_FACTOR:-2}" \
    "${CKPT_ARGS[@]}" \
    --resume \
    "$@"
