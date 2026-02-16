#!/bin/bash
#SBATCH --job-name=pinder_download
#SBATCH --partition=cbio-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err

# Load environment
source /cluster/CBIO/home/vgertner/.bashrc
conda activate atomsurf

# Set path
export REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT

# Define Output Directory
# User requested path
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/pinder-pair"

echo "Downloading and preparing Pinder data to: $DATA_DIR"
echo "This will download ~42k systems for training and all test splits."

# Run preprocess.py
# --test_setting all is now the default, so it will get holo/apo/af2 for test split
# Using 32 workers for parallel processing
python $REPO_ROOT/atomsurf/tasks/pinder_pair/preprocess.py \
    --output_dir $DATA_DIR \
    --test_setting all \
    --num_workers 30

echo "Done!"
