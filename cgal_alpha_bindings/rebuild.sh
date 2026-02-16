#!/bin/bash
# Rebuild CGAL alpha bindings with improved error handling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Rebuilding CGAL Alpha Bindings"
echo "========================================"

# Clean previous build
if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

# Create build directory
mkdir -p build
cd build

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
    echo "✓ Build successful!"
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
