#!/bin/bash
#SBATCH --job-name=time_simp
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.err

# Activate conda environment
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Add cgal_alpha_bindings to PYTHONPATH
export PYTHONPATH="/cluster/CBIO/data2/vgertner/atomsurf/cgal_alpha_bindings/build:$PYTHONPATH"

# Configuration
PDB_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/raw_data_MasifLigand/pdb"
SCRIPT_PATH="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/atomsurf/protein/timing_surface.py"
OUTPUT_CSV="timing_results_simplify_$(date +%Y%m%d_%H%M%S).csv"

echo "=============================================="
echo "Timing Benchmark: Simplify Only (Both Methods)"
echo "=============================================="
echo "Date: $(date)"

# Run Both methods automatically (parameters hardcoded in python script)
echo "--> Running method=both with --simplify..."
python "$SCRIPT_PATH" "$PDB_DIR" \
    --method both \
    --min-verts 256 \
    --simplify \
    --output-csv "$OUTPUT_CSV"

echo "Done."
