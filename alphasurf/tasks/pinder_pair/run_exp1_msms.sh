#!/bin/bash
#SBATCH --job-name=msms_fixed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread

source ~/.bashrc
conda activate atomsurf

python train.py \
    run_name=h100_msms_fixed_dist5_surf2 \
    use_wandb=false \
    data_dir=/lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/atomsurf/data/pinder-pair \
    hydra.searchpath=[file:///lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf/atomsurf/atomsurf/tasks/shared_conf] \
    on_fly=null \
    cfg_surface.data_name=surfaces_msms_fr0.1 \
    encoder=pronet_gvpencoder \
    optimizer.lr=0.0005 \
    epochs=30 \
    interface_distance_graph=5.0 \
    interface_distance_surface=2.0 \
    loader.num_workers=8 \
    loader.use_dynamic_batching=true \
    loader.max_atoms_per_batch=40000 \
    surface_neg_to_pos_ratio=10.0
