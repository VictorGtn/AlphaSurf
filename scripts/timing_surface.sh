#!/bin/bash
#SBATCH --job-name=timing_surface
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/%x_%j.err

# Activate conda environment
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Add cgal_alpha_bindings to PYTHONPATH
export PYTHONPATH="/cluster/CBIO/data2/vgertner/atomsurf/cgal_alpha_bindings/build:$PYTHONPATH"

# --- Configuration ---
# Directory containing PDB files to benchmark
PDB_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/raw_data_MasifLigand/pdb"

# Which method to run: "alpha", "msms", or "both"
METHOD="both"

# Alpha value for alpha complex (used if METHOD is "alpha" or "both")
ALPHA_VALUE=0.0

# Minimum number of vertices for MSMS (used if METHOD is "msms" or "both")
MIN_VERTS=256

# Output CSV file for detailed results
OUTPUT_CSV="timing_results_$(date +%Y%m%d_%H%M%S).csv"

# --- Script Execution ---
# Absolute path to script
SCRIPT_PATH="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/atomsurf/protein/timing_surface.py"

echo "=============================================="
echo "Surface Generation Timing Benchmark"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  PDB directory: $PDB_DIR"
echo "  Method: $METHOD"
echo "  Alpha value: $ALPHA_VALUE"
echo "  Min vertices (MSMS): $MIN_VERTS"
echo "  Output CSV: $OUTPUT_CSV"
echo ""
echo "SLURM settings:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE"
echo ""

# Run timing benchmark with absolute path
python "$SCRIPT_PATH" "$PDB_DIR" \
    --alpha-value $ALPHA_VALUE \
    --min-verts $MIN_VERTS \
    --method $METHOD \
    --output-csv "$OUTPUT_CSV"

echo ""
echo "Timing benchmark completed!"
echo "Results saved to: $OUTPUT_CSV"
