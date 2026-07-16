#!/bin/bash
#SBATCH --job-name=proteingym_alpha
#SBATCH -A pyg@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=21
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=/dev/null
#SBATCH --error=log/proteingym_jz/%x_%j.err
#SBATCH --hint=nomultithread

set -euo pipefail
: "${CKPT:?Set CKPT to the S3F AlphaSurf .ckpt file}"

module purge
module load arch/h100
module load anaconda-py3

export SCRATCH=${SCRATCH:-/lustre/fsn1/projects/rech/pyg/ust26qt}
export WORK=${WORK:-/lustre/fswork/projects/rech/pyg/ust26qt}

eval "$(conda shell.bash hook)"
conda activate "$WORK/atomsurf_h100_new2"

export TMPDIR=${TMPDIR:-$SCRATCH/tmp}
export PYKEOPS_CACHE_DIR=${PYKEOPS_CACHE_DIR:-$SCRATCH/keops_cache}
mkdir -p "$TMPDIR" "$PYKEOPS_CACHE_DIR"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export TOKENIZERS_PARALLELISM=false

REPO_ROOT=$SCRATCH/alphasurf/alphasurf/alphasurf
CGAL_BINDINGS_DIR=$REPO_ROOT/../cgal_alpha_bindings/build
export PYTHONPATH=${PYTHONPATH:-}:$REPO_ROOT/..:$CGAL_BINDINGS_DIR

PROTEINGYM_DIR=${PROTEINGYM_DIR:-$REPO_ROOT/../data/proteingym}
SUBSTITUTIONS_DIR=${SUBSTITUTIONS_DIR:-$PROTEINGYM_DIR/substitutions/DMS_ProteinGym_substitutions}
AF2_DIR=${AF2_DIR:-$PROTEINGYM_DIR/af2_structures/ProteinGym_AF2_structures}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/tasks/proteingym/runs/${SLURM_JOB_ID}}
LOG_DIR=$REPO_ROOT/tasks/proteingym/log
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" log/proteingym_jz

exec >"$LOG_DIR/proteingym_alpha_${SLURM_JOB_ID}.out"

test -f "$CKPT"
test -d "$SUBSTITUTIONS_DIR"
test -d "$AF2_DIR"

cd "$REPO_ROOT"

EXTRA_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    EXTRA_ARGS+=(--limit "$LIMIT")
fi

python -m alphasurf.tasks.proteingym.evaluate \
    --ckpt "$CKPT" \
    --scoring-method option_f \
    --substitutions-dir "$SUBSTITUTIONS_DIR" \
    --af2-dir "$AF2_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "${BATCH_SIZE:-4}" \
    "${EXTRA_ARGS[@]}"
