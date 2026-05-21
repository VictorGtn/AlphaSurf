#!/usr/bin/env python
"""Build script for the curvature_ext C++ extension.

Usage:
    python build.py
"""

import os
import subprocess
import sys
import sysconfig


def main():
    src = os.path.join(os.path.dirname(__file__), "curvature.cpp")

    pybind11_inc = _get_pybind11_include()
    python_inc = sysconfig.get_path("include")
    eigen_inc = _find_eigen()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    outdir = os.path.dirname(__file__)
    outfile = os.path.join(outdir, "curvature_ext" + ext_suffix)

    cmd = [
        "g++",
        "-O3",
        "-shared",
        "-std=c++17",
        "-fPIC",
        f"-I{pybind11_inc}",
        f"-I{python_inc}",
        f"-I{eigen_inc}",
        src,
        "-o",
        outfile,
    ]

    print("Compiling curvature_ext ...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done: {outfile}")


def _get_pybind11_include():
    try:
        import pybind11

        return pybind11.get_include()
    except ImportError:
        raise RuntimeError("pybind11 not installed. Run: pip install pybind11")


def _find_eigen():
    # Check conda env first
    prefix = os.environ.get("CONDA_PREFIX", "")
    if prefix:
        candidate = os.path.join(prefix, "include", "eigen3")
        if os.path.isfile(os.path.join(candidate, "Eigen", "Dense")):
            return candidate

    # Check virtualenv / system paths
    for candidate in [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        os.path.join(sys.prefix, "include", "eigen3"),
    ]:
        if os.path.isfile(os.path.join(candidate, "Eigen", "Dense")):
            return candidate

    raise RuntimeError("Eigen headers not found. Install via: conda install eigen")


if __name__ == "__main__":
    main()
