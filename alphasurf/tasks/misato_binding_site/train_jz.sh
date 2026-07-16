#!/bin/bash
#SBATCH --job-name=misato_jz
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
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append
#SBATCH --output=/lustre/fsn1/projects/rech/pyg/ust26qt/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/pyg/ust26qt/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.err

set -euo pipefail

module purge
module load arch/h100
module load anaconda-py3

export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/pyg/ust26qt}
export WORK=${WORK:-/lustre/fswork/projects/rech/pyg/ust26qt}
export REPO_ROOT=$SCRATCH/alphasurf/alphasurf/alphasurf
export DATA_DIR=${DATA_DIR:-$REPO_ROOT/../data/misato}
export CGAL_BINDINGS_DIR=$REPO_ROOT/../cgal_alpha_bindings/build

eval "$(conda shell.bash hook)"
conda activate "$WORK/atomsurf_h100_new2"
ulimit -n 65536

export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export PYKEOPS_CACHE_DIR=${PYKEOPS_CACHE_DIR:-$SCRATCH/keops_cache}
mkdir -p "$TMPDIR" "$PYKEOPS_CACHE_DIR"

export PYTHONPATH="$REPO_ROOT/..:$CGAL_BINDINGS_DIR:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

LOG_DIR=$REPO_ROOT/tasks/misato_binding_site/log
mkdir -p "$LOG_DIR"

# Refuse to start from a partial rsync transfer.
MD_PATH=$DATA_DIR/MD.hdf5
EXPECTED_MD_BYTES=132841014019
if [ ! -f "$MD_PATH" ]; then
    echo "Missing $MD_PATH" >&2
    exit 1
fi
ACTUAL_MD_BYTES=$(stat -c %s "$MD_PATH")
if [ "$ACTUAL_MD_BYTES" -ne "$EXPECTED_MD_BYTES" ]; then
    echo "Incomplete MD.hdf5: $ACTUAL_MD_BYTES/$EXPECTED_MD_BYTES bytes" >&2
    exit 1
fi
for split in train val test; do
    if [ ! -f "$DATA_DIR/splits/$split.txt" ]; then
        echo "Missing split file: $DATA_DIR/splits/$split.txt" >&2
        exit 1
    fi
done

export ATOMSURF_VERSION=${EXPERIMENT_ID:-${SLURM_JOB_ID:-$(date +%s)}}
RUN_NAME=${RUN_NAME:-misato_random_frame_bs${BATCH_SIZE:-64}}

python -c "import sys; sys.path.insert(0, '$CGAL_BINDINGS_DIR'); import cgal_alpha_algo2; print('[CGAL OK]', cgal_alpha_algo2.__file__)"

CKPT_ARGS=()
if [ -n "${CKPT:-}" ]; then
    CKPT_ARGS+=("ckpt_path=$CKPT")
fi

cd "$REPO_ROOT/.."
srun --export=ALL "$CONDA_PREFIX/bin/python" -m alphasurf.tasks.misato_binding_site.train \
    device=0 \
    data_dir="$DATA_DIR" \
    md_path="$MD_PATH" \
    split_dir="$DATA_DIR/splits" \
    log_dir="$LOG_DIR" \
    hydra.searchpath="[file://$REPO_ROOT/tasks/shared_conf]" \
    run_name="$RUN_NAME" \
    epochs="${EPOCHS:-200}" \
    train_frame_mode="${TRAIN_FRAME_MODE:-random}" \
    loader.batch_size="${BATCH_SIZE:-64}" \
    loader.num_workers="${NUM_WORKERS:-20}" \
    loader.persistent_workers="${PERSISTENT_WORKERS:-true}" \
    loader.prefetch_factor="${PREFETCH_FACTOR:-2}" \
    profile_gpu_memory="${PROFILE_GPU_MEMORY:-false}" \
    "${CKPT_ARGS[@]}" \
    "$@"
