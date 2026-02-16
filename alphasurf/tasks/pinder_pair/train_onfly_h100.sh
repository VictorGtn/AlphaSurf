#!/bin/bash
#SBATCH --job-name=pinder_h100
#SBATCH -A pyg@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_h100-t3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --hint=nomultithread

# 1. Load H100 Environment
module purge
module load arch/h100
module load anaconda-py3

# 2. Activate Conda (NEW H100 ENVIRONMENT)
eval "$(conda shell.bash hook)"
conda activate $SCRATCH/atomsurf_h100_env

# 2.5. Redirect temp to SCRATCH (has much more space)
export TMPDIR=$SCRATCH/tmp
mkdir -p $TMPDIR
export TEMP=$TMPDIR
export TMP=$TMPDIR

# Redirect KeOps cache to SCRATCH (avoid HOME quota)
export PYKEOPS_CACHE_DIR=$SCRATCH/keops_cache
mkdir -p $PYKEOPS_CACHE_DIR

# 3. Exports
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export WANDB_MODE=offline
#export WANDB_DIR=$(pwd)/wandb_logs
#mkdir -p $WANDB_DIR

# Enable segfault debugging (set to empty to disable)
export CRASH_DEBUG_DIR=$(pwd)/crash_debug
mkdir -p $CRASH_DEBUG_DIR

# Force version name to be SLURM_JOB_ID for simple restarts
export ATOMSURF_VERSION=${SLURM_JOB_ID:-debug_$(date +%s)}

export REPO_ROOT=$SCRATCH/atomsurf_h100/atomsurf
export CGAL_BINDINGS_DIR=$REPO_ROOT/cgal_alpha_bindings/build
export PYTHONPATH=$PYTHONPATH:$REPO_ROOT:$CGAL_BINDINGS_DIR

# Debug: Test CGAL is accessible
echo "========================================"
echo "Testing CGAL bindings..."
echo "========================================"
python -c "import sys; sys.path.insert(0, '$CGAL_BINDINGS_DIR'); import cgal_alpha; print('[CGAL] Successfully loaded:', cgal_alpha.__file__)"
echo ""

# Logic to switch between Disk (MSMS) and OnFly (Alpha)
if [ "$MODE" == "disk" ]; then
    ON_FLY_ARGS="on_fly=null cfg_surface.data_name=surfaces_msms_fr0.1"
else
    ON_FLY_ARGS="on_fly.surface_method=${SURFACE_METHOD:-alpha_complex} \
                 on_fly.alpha_value=${ALPHA_VALUE:-0} \
                 on_fly.noise_mode=${NOISE_MODE:-none} \
                 on_fly.sigma_graph=${SIGMA_GRAPH:-0.3} \
                 on_fly.sigma_mesh=${SIGMA_MESH:-0.3} \
                 on_fly.clip_sigma=${CLIP_SIGMA:-null} \
                 on_fly.min_vert_number=16"
fi

# Function to run training with restart on failure
run_training() {
    MAX_RESTARTS=20
    RESTART_COUNT=0
    
    while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
        echo "=== Training attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS ==="
        echo "=== Started at $(date) ==="
        
        # Prepare checkpoint argument (quoted to handle '=' in filenames)
        CKPT_ARG=""
        if [ -n "$CKPT_PATH" ]; then
            CKPT_ARG="+ckpt_path='$CKPT_PATH'"
        fi

        # Run training with auto-resume (added --resume flag)
        python -X faulthandler $REPO_ROOT/atomsurf/tasks/pinder_pair/train.py \
            run_name=${RUN_NAME:-exp_h100} \
            use_wandb=true \
            data_dir=$SCRATCH/atomsurf_h100/atomsurf/data/pinder-pair \
            hydra.searchpath=[file://$SCRATCH/atomsurf_h100/atomsurf/atomsurf/tasks/shared_conf] \
            $ON_FLY_ARGS \
            encoder=pronet_gvpencoder \
            optimizer.lr=0.0005 \
            epochs=${EPOCHS:-30} \
            interface_distance_graph=5.0 \
            interface_distance_surface=${INTERFACE_DIST_SURF:-3.8} \
            loader.num_workers=${NUM_WORKERS:-16} \
            loader.use_dynamic_batching=true \
            loader.max_atoms_per_batch=${MAX_ATOMS:-40000} \
            loader.persistent_workers=${PERSISTENT_WORKERS:-true} \
            surface_neg_to_pos_ratio=10.0 \
            train.save_top_k=5 \
            train.save_top_k=5 \
            ${CKPT_ARG} \
            --resume

        EXIT_CODE=$?
        
        echo "=== Process exited with code $EXIT_CODE at $(date) ==="
        
        # Exit code 0 = success
        if [ $EXIT_CODE -eq 0 ]; then
            echo "Training completed successfully!"
            return 0
        fi
        
        # Negative exit code = killed by signal (e.g., -11 = SIGSEGV)
        # Bash sees 128+signal for killed processes. SIGSEGV=11 -> 139
        if [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -lt 0 ] || [ $EXIT_CODE -eq 1 ]; then
            echo "Process crashed (likely segfault). Restarting in 5 seconds..."
            RESTART_COUNT=$((RESTART_COUNT + 1))
            sleep 5
        else
            # Other error - don't restart
            echo "Process failed with error code $EXIT_CODE. Not restarting."
            return $EXIT_CODE
        fi
    done
    
    echo "Failed after $MAX_RESTARTS restarts. Giving up."
    return 1
}

# Run training with restart logic
run_training
