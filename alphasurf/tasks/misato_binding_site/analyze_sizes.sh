#!/bin/bash
#SBATCH --job-name=misato_sizes
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --output=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf/alphasurf/tasks/misato_binding_site/log/%x_%j.err

set -euo pipefail
source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf

REPO_ROOT=/cluster/CBIO/data2/vgertner/alphasurf/alphasurf
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
mkdir -p "$REPO_ROOT/alphasurf/tasks/misato_binding_site/log"

cd "$REPO_ROOT"
python -m alphasurf.tasks.misato_binding_site.analyze_sizes \
    --data-dir "$REPO_ROOT/data/misato" \
    --num-workers "${SLURM_CPUS_PER_TASK:-20}" \
    "$@"
