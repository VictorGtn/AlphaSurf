#!/bin/bash
#SBATCH --job-name=s3f_precompute
#SBATCH --partition=cbio-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/alphasurf/tasks/s3f_pretrain/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/alphasurf/tasks/s3f_pretrain/log/%x_%j.err
#SBATCH --mem=48000

set +e

source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf

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

PDB_DIR=${PDB_DIR:-/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/data/cath/dompdb}
OUTPUT_DIR=${OUTPUT_DIR:-/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/data/cath/s3f_exact_precomputed}
LIMIT=${LIMIT:-}
OVERWRITE=${OVERWRITE:-}

mkdir -p "$OUTPUT_DIR"

ARGS="--pdb_dir $PDB_DIR --output_dir $OUTPUT_DIR --device cuda"
if [ -n "$LIMIT" ]; then
    ARGS="$ARGS --limit $LIMIT"
fi
if [ -n "$OVERWRITE" ]; then
    ARGS="$ARGS --overwrite"
fi

python -m alphasurf.tasks.s3f_pretrain.precompute_s3f_exact $ARGS
