#!/bin/bash
#SBATCH --job-name=misato_binding_site
#SBATCH --partition=cbio-gpu
#SBATCH --nodelist=node006
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:1
#SBATCH --time=240:00:00
#SBATCH --mem=64000
#SBATCH --output=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.err

set -euo pipefail
source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${SLURM_JOB_GPUS:-${GPU_DEVICE_ORDINAL:-0}}
export KEOS_CACHE_FOLDER=/tmp/keos_cache_${SLURM_JOB_ID}
mkdir -p "$KEOS_CACHE_FOLDER"

REPO_ROOT=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/cgal_alpha_bindings/build:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH:-}"
LOG_DIR="$REPO_ROOT/alphasurf/tasks/misato_binding_site/log"
mkdir -p "$LOG_DIR"
export ATOMSURF_VERSION=${SLURM_JOB_ID:-misato_site_$(date +%s)}

python -m alphasurf.tasks.misato_binding_site.train \
    device=0 log_dir="$LOG_DIR" "$@"
