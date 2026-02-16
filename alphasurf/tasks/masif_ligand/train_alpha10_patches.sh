#!/bin/bash
#SBATCH --job-name=atomsurf_alpha10_patches
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha10_patches_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha10_patches_%j.err
#SBATCH --time=2-00:00:00        
#SBATCH --mem=32000              
#SBATCH --gres=gpu:1             
#SBATCH -p cbio-gpu              
#SBATCH --cpus-per-task=4        
module load python/3.10.1
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

DATA_DIR_NAME="alpha10"
SURFACE_DATA_NAME="surfaces_patches"
RGRAPH_DATA_NAME="rgraph_patches"
RELATIVE_DATA_DIR="../../../data/masif_ligand/$DATA_DIR_NAME"

echo "Launching training for $DATA_DIR_NAME with patches"

python3 train.py \
  data_dir=$RELATIVE_DATA_DIR \
  cfg_surface.use_whole_surfaces=True \
  cfg_surface.data_name=$SURFACE_DATA_NAME \
  cfg_surface.data_dir=$RELATIVE_DATA_DIR \
  cfg_graph.data_name=$RGRAPH_DATA_NAME
