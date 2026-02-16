#!/bin/bash
#SBATCH --job-name=atomsurf_alpha1_patches
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha1_patches_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha1_patches_%j.err
#SBATCH --time=2-00:00:00        
#SBATCH --mem=32000              
#SBATCH --gres=gpu:1             
#SBATCH -p cbio-gpu              
#SBATCH --cpus-per-task=4        
#SBATCH --nodelist=node4        

module load python/3.10.1
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Change to script directory to ensure relative paths work
cd /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/atomsurf/tasks/masif_ligand

# Data directory configuration
DATA_DIR_NAME="alpha1"
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/$DATA_DIR_NAME"
SURFACE_DATA_NAME="surfaces_patches"
RGRAPH_DATA_NAME="rgraph"

echo "Launching training for $DATA_DIR_NAME with patches"
echo "Data directory: $DATA_DIR"
echo "Surface data: $SURFACE_DATA_NAME"
echo "Graph data: $RGRAPH_DATA_NAME"

python3 train.py \
  data_dir=$DATA_DIR \
  cfg_surface.use_whole_surfaces=False \
  cfg_surface.data_name=$SURFACE_DATA_NAME \
  cfg_surface.data_dir=$DATA_DIR \
  cfg_graph.data_name=$RGRAPH_DATA_NAME \
  cfg_graph.data_dir=$DATA_DIR
