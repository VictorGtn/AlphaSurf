# CGAL Alpha Complex Python Bindings

Python bindings for CGAL's `Fixed_alpha_shape_3`.

## Dependencies

- CGAL 5.x+
- GMP, MPFR
- pybind11
- Python 3.8+
- CMake 3.16+

## Build

### Linux (Ubuntu/Debian)

```bash
sudo apt install libcgal-dev libgmp-dev libmpfr-dev python3-dev cmake
pip install pybind11

mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS (Homebrew)

```bash
brew install cgal gmp mpfr
pip install pybind11

mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Conda

```bash
conda install -c conda-forge cgal-cpp pybind11

mkdir build && cd build
cmake ..
make -j8
```

## Spherical-patch graph binding

`cgal_patch_graph` is a separate extension which returns one graph node per
connected exterior spherical patch, exact SBL patch areas, and total shared
boundary-arc lengths:

Patch-area and circular-arc reconstruction use SBL's exact spherical kernel.
This avoids inconsistent three-sphere intersections observed with its
double-precision spherical kernel on full molecular assemblies, at the cost of
slower extraction. Cache the full-protein patch graph when several pockets use
the same protein.

```bash
cmake --build build --target cgal_patch_graph -j8
```

```python
import cgal_patch_graph

geometry = cgal_patch_graph.compute_patch_graph_from_atoms(
    positions, radii, alpha=0.0, probe_radius=1.4
)
```

The returned `edge_index` contains each undirected patch pair once. Use
`alphasurf.protein.patch_operators` to construct the sparse mass/stiffness
matrices and generalized eigenbasis. The initial integration uses DiffusionNet
with `with_gradient_features=False`.
