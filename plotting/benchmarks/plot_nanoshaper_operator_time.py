#!/usr/bin/env python3
"""
Plot NanoShaper Spectral Operator Computation Time vs Grid Scale.

Measures the time taken to compute spectral operators (Laplacian + eigendecomp)
on NanoShaper generated surfaces across different grid scales.
"""

import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = str(Path(__file__).resolve().parents[2])
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "benchmarks"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alphasurf.protein.create_operators import compute_operators
from alphasurf.protein.graphs import parse_pdb_path

NANOSHAPER_BIN = str(
    Path(project_root).parent / "nanoshaper-master" / "build" / "NanoShaper"
)
PDB_DIR = os.path.join(project_root, "data", "pinder-pair", "pdb")


def _parse_off(off_path):
    with open(off_path, "r") as f:
        line = f.readline()
        if line.startswith("OFF"):
            line = f.readline()
        while line.startswith("#") or line.strip() == "":
            line = f.readline()
        n_verts, n_faces, _ = map(int, line.split())
        verts = np.loadtxt(
            f, max_rows=n_verts, dtype=np.float32, usecols=(0, 1, 2), ndmin=2
        )
        faces = np.loadtxt(
            f, max_rows=n_faces, dtype=np.int32, usecols=(1, 2, 3), ndmin=2
        )
    return verts, faces


def try_nanoshaper(pdb_path, grid_scale=0.5, probe_radius=1.4):
    parsed = parse_pdb_path(str(pdb_path), use_pqr=False)
    if parsed is None or parsed[5] is None or parsed[7] is None:
        return None, "parse_failed"
    atom_pos, atom_rad = parsed[5], parsed[7]

    work_dir = tempfile.mkdtemp(prefix="ns_plot_")
    xyzr_file = os.path.join(work_dir, "atoms.xyzr")
    conf_file = os.path.join(work_dir, "conf.prm")
    off_file = os.path.join(work_dir, "triangulatedSurf.off")

    try:
        with open(xyzr_file, "w") as f:
            for i in range(len(atom_pos)):
                f.write(
                    f"{atom_pos[i, 0]:.6f} {atom_pos[i, 1]:.6f} {atom_pos[i, 2]:.6f} {atom_rad[i]:.6f}\n"
                )
        with open(conf_file, "w") as f:
            f.write(
                f"Compute_Vertex_Normals = true\n"
                f"Save_Mesh_MSMS_Format = false\n"
                f"Load_Balancing = true\n"
                f"Grid_scale = {grid_scale}\n"
                f"Grid_perfil = 80.0\n"
                f"XYZR_FileName = {xyzr_file}\n"
                f"Build_epsilon_maps = false\n"
                f"Build_status_map = true\n"
                f"Tri2Balls = false\n"
                f"Surface = ses\n"
                f"Smooth_Mesh = true\n"
                f"Number_thread = 1\n"
                f"Skin_Surface_Parameter = 0.45\n"
                f"Blobbyness = -2.5\n"
                f"Skip_Mem_CleanUp = true\n"
                f"Patch_Based_Algorithm = true\n"
                f"Analytical_Ray_Vs_Torus_Intersection = true\n"
                f"Force_Serial_Build = false\n"
                f"Max_Num_Atoms = -1\n"
                f"Domain_Shrinkage = 1.0\n"
                f"Optimize_Grids = true\n"
                f"Cavity_Detection_Filling = true\n"
                f"Conditional_Volume_Filling_Value = 99999.0\n"
                f"Keep_Water_Shaped_Cavities = false\n"
                f"Probe_Radius = {probe_radius}\n"
                f"Max_Probes_Self_Intersections = 100\n"
                f"Self_Intersections_Grid_Coefficient = 1.5\n"
                f"Accurate_Triangulation = true\n"
                f"Triangulation = true\n"
                f"Check_duplicated_vertices = true\n"
                f"Save_Status_map = false\n"
                f"Save_PovRay = false\n"
            )
        result = subprocess.run(
            [NANOSHAPER_BIN, conf_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not os.path.exists(off_file):
            return None, f"no_output(exit={result.returncode})"
        verts, faces = _parse_off(off_file)
        return (verts, faces), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as e:
        return None, str(e)[:80]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _timeout_handler(signum, frame):
    raise TimeoutError("process_one timed out")


def process_one(args):
    import signal

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(300)

    pdb_path, grid_scale = args
    name = Path(pdb_path).stem

    try:
        ns_crash = False
        ns_verts = 0
        spectral_time = 0.0

        res, err = try_nanoshaper(pdb_path, grid_scale=grid_scale)
        if err:
            ns_crash = True
        else:
            verts, faces = res
            ns_verts = len(verts)

            if len(faces) > 0 and ns_verts > 128:
                t0 = time.time()
                try:
                    compute_operators(
                        verts,
                        faces,
                        k_eig=128,
                        use_fem_decomp=False,
                        use_robust_laplacian=False,
                    )
                    spectral_time = time.time() - t0
                except Exception:
                    spectral_time = -1.0

    except TimeoutError:
        ns_crash = True
        ns_verts = 0
        spectral_time = 0.0
    finally:
        signal.alarm(0)

    return {
        "pdb": name,
        "grid_scale": grid_scale,
        "ns_crash": ns_crash,
        "ns_verts": ns_verts,
        "spectral_time": spectral_time,
    }


def _save_csvs(results_by_scale):
    """Save raw and summary CSVs from completed grid scales."""
    import csv

    all_raw = []
    for gs in sorted(results_by_scale.keys()):
        all_raw.extend(results_by_scale[gs])
    if not all_raw:
        return

    raw_csv = FIG_DIR / "nanoshaper_operator_time_raw.csv"
    fields = list(all_raw[0].keys())
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_raw)

    summary_csv = FIG_DIR / "nanoshaper_operator_time_summary.csv"
    rows = []
    for gs in sorted(results_by_scale.keys()):
        rs = results_by_scale[gs]
        valid_spectral = [
            r["spectral_time"]
            for r in rs
            if r["spectral_time"] > 0 and not r["ns_crash"]
        ]
        valid_verts = [r["ns_verts"] for r in rs if not r["ns_crash"]]
        avg_spectral = np.mean(valid_spectral) if valid_spectral else 0
        avg_verts = np.mean(valid_verts) if valid_verts else 0
        rows.append([gs, avg_verts, avg_spectral])
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Grid Scale", "Avg Vertices", "Avg Spectral Time (s)"])
        writer.writerows(rows)

    print(f"  Saved {raw_csv} + {summary_csv} ({len(all_raw)} rows)", flush=True)

    # Print completed scales so far
    print("  --- Completed grid scales ---")
    print(
        f"  {'Grid Scale':>10} | {'Avg Verts':>12} | {'Avg Spectral (s)':>18} | {'N valid':>8}"
    )
    for gs, avg_verts, avg_spectral in rows:
        rs = results_by_scale[gs]
        n_valid = sum(1 for r in rs if r["spectral_time"] > 0 and not r["ns_crash"])
        print(f"  {gs:10.2f} | {avg_verts:12.1f} | {avg_spectral:18.3f} | {n_valid:8d}")
    print(flush=True)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10, help="Number of workers")
    parser.add_argument(
        "--max-files", type=int, default=None, help="Max PDB files to process"
    )
    parser.add_argument(
        "--grid-scales",
        type=str,
        default="0.3,0.4,0.5,0.6,0.8,1.0",
        help="Comma-separated grid scales",
    )
    parser.add_argument(
        "--output-img",
        type=str,
        default="nanoshaper_operator_time.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

    # Ensure CWD exists (spawn context needs os.getcwd() to work)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Skip unsuffixed PDBs that have a _holo counterpart — those are unused test structures
    holo_stems = {p.stem.replace("_holo", "") for p in Path(PDB_DIR).glob("*_holo.pdb")}
    pdb_files = sorted(
        str(p) for p in Path(PDB_DIR).glob("*.pdb") if p.stem not in holo_stems
    )
    if args.max_files:
        pdb_files = pdb_files[: args.max_files]

    grid_scales = [float(x.strip()) for x in args.grid_scales.split(",")]

    results_by_scale = {}
    ctx = multiprocessing.get_context("spawn")

    wall_clock = {}
    for gs in grid_scales:
        print(
            f"\n  NanoShaper gs={gs} ({len(pdb_files)} tasks, {args.workers} workers)"
        )
        task_args = [(p, gs) for p in pdb_files]
        n = len(task_args)
        gs_results = []

        t_wall_start = time.time()
        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_one, task_args), 1):
                gs_results.append(r)
                if i % 50 == 0:
                    print(
                        f"    {i}/{n} done ({time.time() - t_wall_start:.1f}s)",
                        flush=True,
                    )
        wall_clock[gs] = time.time() - t_wall_start
        print(f"    {n}/{n} done -- wall clock: {wall_clock[gs]:.1f}s", flush=True)

        results_by_scale[gs] = gs_results
        _save_csvs(results_by_scale)

    x_scales = []
    y_spectral_time = []
    y_vertices = []

    for gs in sorted(results_by_scale.keys()):
        rs = results_by_scale[gs]

        valid_spectral = [
            r["spectral_time"]
            for r in rs
            if r["spectral_time"] > 0 and not r["ns_crash"]
        ]
        valid_verts = [r["ns_verts"] for r in rs if not r["ns_crash"]]

        avg_spectral = np.mean(valid_spectral) if valid_spectral else 0
        avg_verts = np.mean(valid_verts) if valid_verts else 0

        x_scales.append(gs)
        y_spectral_time.append(avg_spectral)
        y_vertices.append(avg_verts)

    print("\n" + "=" * 100)
    print(
        f"{'Grid Scale':>10} | {'Avg Verts':>12} | {'Avg Spectral (s)':>18} | "
        f"{'N valid':>8} | {'Wall Clock':>11} | {'Time/Prot':>11}"
    )
    print("-" * 100)
    for gs, verts, spec in zip(x_scales, y_vertices, y_spectral_time):
        rs = results_by_scale[gs]
        n_valid = sum(1 for r in rs if r["spectral_time"] > 0 and not r["ns_crash"])
        wc = wall_clock.get(gs, 0.0)
        t_per_prot = wc / n_valid if n_valid > 0 else 0.0
        print(
            f"{gs:10.2f} | {verts:12.1f} | {spec:18.3f} | "
            f"{n_valid:8d} | {wc:9.1f}s | {t_per_prot:9.4f}s"
        )
    print("=" * 100)
    total_wall = sum(wall_clock.values())
    total_valid = sum(
        sum(
            1
            for r in results_by_scale[gs]
            if r["spectral_time"] > 0 and not r["ns_crash"]
        )
        for gs in results_by_scale
    )
    print(f"Total wall clock: {total_wall:.1f}s ({total_wall / 60:.1f}min)")
    print(f"Total valid proteins (all scales): {total_valid}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:orange"
    ax1.set_xlabel("Grid Scale", fontsize=12)
    ax1.set_ylabel("Spectral Operator Time (seconds)", color=color1, fontsize=12)
    line1 = ax1.plot(
        x_scales, y_spectral_time, marker="o", color=color1, label="Spectral Time"
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Average Number of Vertices", color=color2, fontsize=12)
    line2 = ax2.plot(
        x_scales,
        y_vertices,
        marker="s",
        linestyle="--",
        color=color2,
        label="Avg Vertices",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper center")

    plt.title("NanoShaper: Spectral Operator Time vs Grid Scale", fontsize=14)
    plt.tight_layout()

    out_path = FIG_DIR / args.output_img
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
