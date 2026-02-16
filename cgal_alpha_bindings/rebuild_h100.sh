#!/bin/bash
# Rebuild CGAL alpha bindings for H100 architecture

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Rebuilding CGAL Alpha Bindings for H100"
echo "========================================"

# Load H100 architecture
module purge
module load arch/h100
module load anaconda-py3

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /lustre/fsn1/projects/rech/pyg/ust26qt/atomsurf_env

# Clean previous H100 build
if [ -d "build_h100" ]; then
    echo "Cleaning previous H100 build..."
    rm -rf build_h100
fi

# Create build directory
mkdir -p build_h100
cd build_h100

# Configure with CMake
echo "Running CMake..."
cmake ..

# Build
echo "Building..."
make -j$(nproc)

# Check if successful
SO_FILE=$(ls cgal_alpha*.so 2>/dev/null | head -1)
if [ -n "$SO_FILE" ]; then
    echo ""
    echo "========================================"
    echo "✓ H100 Build successful!"
    echo "========================================"
    echo "Library: $SO_FILE"
    echo ""
    echo "Test import:"
    python3 -c "import sys; sys.path.insert(0, '.'); import cgal_alpha; print('✓ Import successful')"
else
    echo ""
    echo "✗ Build failed - no .so file generated"
    exit 1
fi
