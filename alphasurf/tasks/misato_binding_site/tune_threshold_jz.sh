#!/bin/bash
#SBATCH --job-name=misato_threshold
#SBATCH -A pyg@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --hint=nomultithread
#SBATCH --output=/lustre/fsn1/projects/rech/pyg/ust26qt/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.out
#SBATCH --error=/lustre/fsn1/projects/rech/pyg/ust26qt/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.err

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch $0 /path/to/a.ckpt [/path/to/b.ckpt ...]" >&2
    exit 2
fi

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

checkpoint_list=""
for checkpoint in "$@"; do
    if [ -n "$checkpoint_list" ]; then
        checkpoint_list+=","
    fi
    checkpoint_list+="'$checkpoint'"
done

output_path=$REPO_ROOT/tasks/misato_binding_site/log/threshold_tuning_${SLURM_JOB_ID}.json

cd "$REPO_ROOT/.."
srun --export=ALL "$CONDA_PREFIX/bin/python" \
    -m alphasurf.tasks.misato_binding_site.tune_threshold \
    data_dir="$DATA_DIR" \
    md_path="$DATA_DIR/MD.hdf5" \
    split_dir="$DATA_DIR/splits" \
    hydra.searchpath="[file://$REPO_ROOT/tasks/shared_conf]" \
    loader.batch_size="${BATCH_SIZE:-128}" \
    loader.num_workers="${NUM_WORKERS:-20}" \
    loader.persistent_workers=false \
    loader.prefetch_factor=2 \
    "+threshold_checkpoint_paths=[$checkpoint_list]" \
    +threshold_output="$output_path"
