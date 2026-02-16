#!/bin/bash
#SBATCH --job-name=bench_lap
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/alphasurf/atomsurf/atomsurf/tasks/pinder_pair/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/alphasurf/atomsurf/atomsurf/tasks/pinder_pair/%x_%j.err

# Activate conda environment
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

INPUT_DIR="/cluster/CBIO/data2/vgertner/alphasurf/atomsurf/data/pinder-pair/surfaces/alpha_complex_1.0_a0.0"
LIMIT=3000

echo "Running Laplacian Benchmark on $INPUT_DIR (limit: $LIMIT)"
python /cluster/CBIO/data2/vgertner/alphasurf/atomsurf/scripts/benchmark_laplacian.py "$INPUT_DIR" --limit "$LIMIT"
