#!/bin/bash -l
#SBATCH --job-name=mesh_stats
#SBATCH --output=log/mesh_stats/%x_%j.log
#SBATCH --error=log/mesh_stats/%x_%j.err
#SBATCH --time=2:00:00        
#SBATCH --mem=32000              
#SBATCH -p cbio-cpu       
#SBATCH --cpus-per-task=16

source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

echo "Starting mesh statistics analysis..."
echo "Using $SLURM_CPUS_PER_TASK CPUs"

python3 /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/scripts/mesh_statistics.py \
    --n_workers $SLURM_CPUS_PER_TASK \
    --dirs /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/msms_01/surfaces_full_msms_fr1.0 \
           /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/msms_01/surfaces_full_msms_fr0.1 \
           /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/msms_01/surfaces_full_alpha0_fr1.0 \
           /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/msms_01/surfaces_full_alpha5_fr1.0 \
    --names MSMS MSMS_Coarsened Alpha_0 Alpha_5

echo "Done!"
