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
