#!/bin/bash
#SBATCH --job-name=atomsurf_alpha0_patches_dmasif
#SBATCH --output=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha0_patches_dmasif_%j.log
#SBATCH --error=/cluster/CBIO/data2/vgertner/atomsurf/log/train_alpha0_patches_dmasif_%j.err
#SBATCH --time=2-00:00:00        
#SBATCH --mem=32000              
#SBATCH --gres=gpu:1             
#SBATCH -p cbio-gpu              
#SBATCH --cpus-per-task=4   
#SBATCH --nodelist=node005

module load python/3.10.1
source /cluster/CBIO/home/vgertner/miniconda3/etc/profile.d/conda.sh
conda activate atomsurf

# Help pykeops find CUDA libraries - try multiple approaches
# 1. Find PyTorch's bundled CUDA libraries
PYTORCH_CUDA_LIB=$(python3 -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null)
if [ -d "$PYTORCH_CUDA_LIB" ]; then
    export LD_LIBRARY_PATH=$PYTORCH_CUDA_LIB:$LD_LIBRARY_PATH
    echo "Added PyTorch CUDA libs: $PYTORCH_CUDA_LIB"
fi

# 2. Add conda environment lib directory
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 3. Try to find system CUDA (common locations)
for CUDA_PATH in /usr/local/cuda /opt/cuda /usr/lib/cuda; do
    if [ -d "$CUDA_PATH/lib64" ]; then
        export CUDA_HOME=$CUDA_PATH
        export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
        echo "Found CUDA at: $CUDA_PATH"
        break
    fi
done

# 4. Find CUDA headers (cuda.h) - needed for pykeops compilation
CUDA_INCLUDE_DIR=$(python3 << EOF
import os
import glob

# Search for cuda.h in common locations
search_paths = [
    os.environ.get('CONDA_PREFIX', '') + '/include',
    os.environ.get('CONDA_PREFIX', '') + '/pkgs/cudatoolkit*/include',
    '/usr/local/cuda/include',
    '/opt/cuda/include',
    '/usr/include/cuda',
]

for path in search_paths:
    if path and os.path.exists(path):
        cuda_h = os.path.join(path, 'cuda.h')
        if os.path.exists(cuda_h):
            print(path)
            break
    # Try glob pattern for conda packages
    if 'pkgs' in path:
        matches = glob.glob(path)
        for match in matches:
            cuda_h = os.path.join(match, 'include', 'cuda.h')
            if os.path.exists(cuda_h):
                print(os.path.join(match, 'include'))
                break
EOF
)

if [ -n "$CUDA_INCLUDE_DIR" ]; then
    # Set CUDA_HOME to parent of include directory
    CUDA_HOME_FROM_INCLUDE=$(dirname "$CUDA_INCLUDE_DIR")
    if [ -z "$CUDA_HOME" ] || [ "$CUDA_HOME" != "$CUDA_HOME_FROM_INCLUDE" ]; then
        export CUDA_HOME=$CUDA_HOME_FROM_INCLUDE
        export CUDA_PATH=$CUDA_HOME_FROM_INCLUDE
        echo "Found CUDA headers in: $CUDA_INCLUDE_DIR"
        echo "Set CUDA_HOME=$CUDA_HOME"
    fi
else
    echo "WARNING: CUDA headers (cuda.h) not found!"
    echo "Pykeops compilation may fail. CUDA headers are needed for CUDA kernel compilation."
fi

# 5. Set CUDA_HOME from conda if still not set
if [ -z "$CUDA_HOME" ] && [ -d "$CONDA_PREFIX" ]; then
    # Check if cudatoolkit is installed in conda
    if [ -d "$CONDA_PREFIX/pkgs/cudatoolkit" ]; then
        export CUDA_HOME=$CONDA_PREFIX
        export CUDA_PATH=$CONDA_PREFIX
    fi
fi

cd /cluster/CBIO/data2/vgertner/atomsurf/atomsurf/atomsurf/tasks/masif_ligand

DATA_DIR_NAME="alpha0"
DATA_DIR="/cluster/CBIO/data2/vgertner/atomsurf/atomsurf/data/masif_ligand/$DATA_DIR_NAME"
SURFACE_DATA_NAME="surfaces_patches_geom_areaweighted"
RGRAPH_DATA_NAME="rgraph"

echo "Launching training for $DATA_DIR_NAME with patches"
echo "Data directory: $DATA_DIR"
echo "Surface data: $SURFACE_DATA_NAME"
echo "Graph data: $RGRAPH_DATA_NAME"

# Verify CUDA setup and find libcudart.so
echo "=== CUDA Setup Verification ==="
CUDA_RUNTIME_DIR=$(python3 << EOF
import torch
import os
import glob

torch_lib = os.path.dirname(torch.__file__) + '/lib'
search_paths = [
    torch_lib,
    os.environ.get('CONDA_PREFIX', '') + '/lib',
    '/usr/local/cuda/lib64',
    '/opt/cuda/lib64',
    '/usr/lib/cuda/lib64',
]

for path in search_paths:
    if path and os.path.exists(path):
        matches = glob.glob(os.path.join(path, 'libcudart.so*'))
        if matches:
            print(os.path.dirname(matches[0]))
            break
EOF
)

if [ -n "$CUDA_RUNTIME_DIR" ]; then
    export LD_LIBRARY_PATH=$CUDA_RUNTIME_DIR:$LD_LIBRARY_PATH
    # Set CUDA_HOME to parent directory if not already set
    if [ -z "$CUDA_HOME" ]; then
        export CUDA_HOME=$(dirname "$CUDA_RUNTIME_DIR")
    fi
    echo "Found libcudart.so in: $CUDA_RUNTIME_DIR"
    echo "Added to LD_LIBRARY_PATH and set CUDA_HOME=$CUDA_HOME"
else
    echo "ERROR: libcudart.so not found! Pykeops requires CUDA runtime libraries."
    echo ""
    echo "To fix this, install cudatoolkit in your conda environment:"
    echo "  conda activate atomsurf"
    echo "  conda install cudatoolkit=11.8 -c nvidia"
    echo ""
    echo "Then re-run this script."
    echo ""
    echo "Exiting to prevent segfaults..."
    exit 1
fi

python3 << EOF
import torch
import os
import glob

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
torch_lib = os.path.dirname(torch.__file__) + '/lib'
if os.path.exists(torch_lib):
    cuda_libs = [f for f in os.listdir(torch_lib) if 'cuda' in f.lower()][:5]
    print(f"PyTorch CUDA libraries: {cuda_libs}")

# Check for CUDA headers
cuda_home = os.environ.get('CUDA_HOME', '')
if cuda_home:
    cuda_h = os.path.join(cuda_home, 'include', 'cuda.h')
    if os.path.exists(cuda_h):
        print(f"CUDA headers found: {cuda_h}")
    else:
        print(f"WARNING: CUDA headers not found at {cuda_h}")
        # Try to find them
        search_dirs = [
            os.path.join(cuda_home, 'include'),
            '/usr/local/cuda/include',
            '/opt/cuda/include',
        ]
        for d in search_dirs:
            if os.path.exists(os.path.join(d, 'cuda.h')):
                print(f"Found CUDA headers at: {d}")
                break
EOF
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: ${CUDA_PATH:-not set}"
echo "================================"

# Pykeops configuration - verbose mode to see compilation issues
export PYKEOPS_VERBOSE=1
# Ensure pykeops cache base directory exists with proper permissions
mkdir -p ~/.cache
chmod 755 ~/.cache 2>/dev/null || true
# Pre-create keops cache directory structure to avoid FileNotFoundError
# Pykeops creates subdirectories based on system info, so create a base keops directory
mkdir -p ~/.cache/keops2.3 2>/dev/null || true
# Ensure keops can create its build directories by pre-creating a dummy structure
# This helps avoid FileNotFoundError when pykeops tries to create temp files
python3 << PYEOF
import os
cache_dir = os.path.expanduser("~/.cache")
keops_dir = os.path.join(cache_dir, "keops2.3")
os.makedirs(keops_dir, exist_ok=True, mode=0o755)
# Try to create a test subdirectory to ensure permissions work
test_subdir = os.path.join(keops_dir, "test")
try:
    os.makedirs(test_subdir, exist_ok=True)
    os.rmdir(test_subdir)
    print("Keops cache directory is writable")
except Exception as e:
    print(f"Warning: Keops cache directory may have permission issues: {e}")
PYEOF
# Clear pykeops cache to force recompilation with correct CUDA paths
rm -rf ~/.cache/pykeops* 2>/dev/null
# Don't remove keops directory entirely, just let pykeops recreate subdirectories
echo "Cleared pykeops cache and ensured cache directory exists"

python3 train.py \
  data_dir=$DATA_DIR \
  cfg_surface.use_whole_surfaces=False \
  cfg_surface.data_name=$SURFACE_DATA_NAME \
  cfg_surface.data_dir=$DATA_DIR \
  cfg_graph.use_graphs=true \
  cfg_graph.data_name=$RGRAPH_DATA_NAME \
  cfg_graph.data_dir=$DATA_DIR \
  encoder=dmasif_pronet_gvpencoder \
  dmasif_block.dim_in=128 \
  optimizer.lr=0.0001 \
  scheduler=reduce_lr_on_plateau \
  epochs=2 \
  loader.batch_size=4 \
  loader.num_workers=16 \
  train.save_top_k=5 \
  train.early_stoping_patience=500 \
  run_name=dmasif_pronet_gvp_alpha0_patches \
  exclude_failed_patches=False \
  device=0