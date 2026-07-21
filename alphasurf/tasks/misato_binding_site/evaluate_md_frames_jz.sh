#!/bin/bash
#SBATCH --job-name=misato_md_eval
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
    # Slurm starts the job independently of the shell that submitted it, and
    # this script later changes directory to the repository root. Resolve
    # relative checkpoint arguments against the submission directory first.
    if [[ "$checkpoint" != /* ]]; then
        checkpoint="${SLURM_SUBMIT_DIR:-$PWD}/$checkpoint"
    fi
    if [ ! -f "$checkpoint" ]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 2
    fi
    checkpoint=$(realpath "$checkpoint")
    if [ -n "$checkpoint_list" ]; then
        checkpoint_list+=","
    fi
    checkpoint_list+="'$checkpoint'"
done

fractions=${FRAME_FRACTIONS:-"0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"}
output_path=$REPO_ROOT/tasks/misato_binding_site/log/md_frame_evaluation_${SLURM_JOB_ID}.json

cd "$REPO_ROOT/.."
srun --export=ALL "$CONDA_PREFIX/bin/python" \
    -m alphasurf.tasks.misato_binding_site.evaluate_md_frames \
    data_dir="$DATA_DIR" \
    md_path="$DATA_DIR/MD.hdf5" \
    split_dir="$DATA_DIR/splits" \
    hydra.searchpath="[file://$REPO_ROOT/tasks/shared_conf]" \
    loader.batch_size="${BATCH_SIZE:-128}" \
    loader.num_workers="${NUM_WORKERS:-20}" \
    loader.persistent_workers=false \
    loader.prefetch_factor=2 \
    "+md_eval_checkpoint_paths=[$checkpoint_list]" \
    "+md_eval_frame_fractions=[$fractions]" \
    +md_eval_output="$output_path"
