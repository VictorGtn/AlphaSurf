#!/bin/bash
#SBATCH --job-name=s3f_pretrain
#SBATCH --partition=cbio-gpu
#SBATCH --nodelist=node006
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --time=240:00:00
#SBATCH --output=log/s3f_pretrain/%x_%j.log
#SBATCH --error=log/s3f_pretrain/%x_%j.err
#SBATCH --mem=64000

set -euo pipefail

source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf_poisson2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-0}}
export KEOS_CACHE_FOLDER=/tmp/keos_cache_${SLURM_JOB_ID}
mkdir -p $KEOS_CACHE_FOLDER
echo "=== Assigned GPU: $CUDA_VISIBLE_DEVICES ==="

REPO_ROOT=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$REPO_ROOT/cgal_alpha_bindings/build
export LD_LIBRARY_PATH=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):$LD_LIBRARY_PATH

ulimit -n 65536

cd "$REPO_ROOT"
mkdir -p log/s3f_pretrain
export ATOMSURF_VERSION="${SLURM_JOB_ID:-s3f_$(date +%s)}"

python -m alphasurf.tasks.s3f_pretrain.train \
    device=0 \
    run_name=s3f_pretrain \
    log_dir="$REPO_ROOT/log/s3f_pretrain" \
    "$@"
