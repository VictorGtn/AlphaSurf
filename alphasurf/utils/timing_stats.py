"""
Timing statistics collector for surface preprocessing.
Collects timing data across multiple samples and prints summary stats.
Uses per-process files for multiprocessing compatibility, aggregates at summary time.
"""

import glob
import os
import tempfile
import time

import numpy as np

_TIMING_PREFIX = "alphasurf_timing_"


def _get_timing_dir():
    """Get timing directory, isolated by SLURM job ID if available."""
    base_dir = tempfile.gettempdir()
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id:
        timing_dir = os.path.join(base_dir, f"alphasurf_timing_{job_id}")
        os.makedirs(timing_dir, exist_ok=True)
        return timing_dir
    return base_dir


def _get_timing_file():
    """Get per-process timing file path."""
    return os.path.join(_get_timing_dir(), f"{_TIMING_PREFIX}{os.getpid()}.csv")


def reset():
    """Clear all timing files in the job-specific directory."""
    timing_dir = _get_timing_dir()
    pattern = os.path.join(timing_dir, f"{_TIMING_PREFIX}*.csv")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except:
            pass


def record(name: str, elapsed: float):
    """Append timing entry to per-process file."""
    filepath = _get_timing_file()
    try:
        with open(filepath, "a") as f:
            f.write(f"{name},{elapsed}\n")
    except Exception:
        pass


def print_summary():
    """Aggregate all per-process timing files and print summary."""
    timing_dir = _get_timing_dir()
    pattern = os.path.join(timing_dir, f"{_TIMING_PREFIX}*.csv")
    all_files = glob.glob(pattern)

    if not all_files:
        print("No timing data recorded.")
        return

    # Aggregate all timing data
    data = {}
    for filepath in all_files:
        try:
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or "," not in line:
                        continue
                    try:
                        name, elapsed = line.rsplit(",", 1)
                        elapsed = float(elapsed)
                        if name not in data:
                            data[name] = []
                        data[name].append(elapsed)
                    except ValueError:
                        continue
        except Exception:
            continue

    if not data:
        print("No timing data recorded.")
        return

    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)

    # Print priority metrics first
    for name in [
        "surface_generation",
        "mesh_processing",
        "compute_operators",
        "geom_feats",
    ]:
        if name not in data:
            continue
        t = np.array(data[name])
        n = len(t)
        mean = np.mean(t)
        std = np.std(t)
        total = np.sum(t)
        print(f"{name:20s}: {mean:.4f}s ± {std:.4f}s  (n={n}, total={total:.2f}s)")

    # Print remaining metrics
    for name in sorted(data.keys()):
        if name in [
            "surface_generation",
            "mesh_processing",
            "compute_operators",
            "geom_feats",
        ]:
            continue
        t = np.array(data[name])
        n = len(t)
        mean = np.mean(t)
        std = np.std(t)
        total = np.sum(t)
        print(f"{name:20s}: {mean:.4f}s ± {std:.4f}s  (n={n}, total={total:.2f}s)")

    print("=" * 60 + "\n")


class TimingStats:
    """Wrapper class for backward compatibility."""

    @staticmethod
    def get():
        return TimingStats()

    @staticmethod
    def reset():
        reset()

    @staticmethod
    def print_summary():
        print_summary()


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, stats=None):
        self.name = name
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        record(self.name, self.elapsed)
