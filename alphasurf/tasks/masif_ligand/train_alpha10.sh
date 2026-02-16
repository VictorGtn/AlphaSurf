#!/bin/bash
#SBATCH --job-name=atomsurf_alpha10
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha10_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha10_%j.err
#SBATCH --time=2-00:00:00        
#SBATCH --mem=32000              
#SBATCH --gres=gpu:1             
#SBATCH -p cbio-gpu              
#SBATCH --cpus-per-task=4        
module load python/3.10.1
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Parameters for this run
DATA_DIR_NAME="alpha10"
DATA_NAME_VALUE="surfaces_full_alpha_complex_1.0_False"
RELATIVE_DATA_DIR="../../../data/masif_ligand/$DATA_DIR_NAME"

echo "Launching training for $DATA_DIR_NAME with $DATA_NAME_VALUE"

python3 train.py \
  data_dir=$RELATIVE_DATA_DIR \
  cfg_surface.use_whole_surfaces=True \
  cfg_surface.data_name=$DATA_NAME_VALUE \
  cfg_surface.data_dir=$RELATIVE_DATA_DIR
