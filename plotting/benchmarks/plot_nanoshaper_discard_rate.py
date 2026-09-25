#!/usr/bin/env python3
"""
Plot NanoShaper Discard Rate vs. Average Vertices across Grid Scales.

Computes the number of surfaces discarded due to multiple connected components
(>1 component larger than 1% of the maximum component) and the average number
of vertices, plotted against the grid scale.

Uses multiprocessing to evaluate NanoShaper across a range of grid scales.
"""

import argparse
import csv
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

project_root = str(Path(__file__).resolve().parents[2])
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "benchmarks"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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


def count_components(verts, faces):
    if len(faces) == 0:
        return 0, len(verts), 0, False
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.remove_degenerate_triangles()
    clusters, cluster_n, _ = mesh.cluster_connected_triangles()
    cluster_n = np.asarray(cluster_n)
    n_comps = len(cluster_n)
    largest = int(cluster_n.max()) if len(cluster_n) > 0 else 0
    cutoff = max(1, int(0.01 * largest))
    n_above_cutoff = int((cluster_n >= cutoff).sum())
    multi_comp = n_above_cutoff > 1
    return n_comps, len(verts), len(faces), multi_comp


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
    signal.alarm(180)

    pdb_path, grid_scale = args
    name = Path(pdb_path).stem

    try:
        ns_crash = False
        ns_multi_comp = False
        ns_verts = 0

        res, err = try_nanoshaper(pdb_path, grid_scale=grid_scale)
        if err:
            ns_crash = True
        else:
            _, ns_verts, _, ns_multi_comp = count_components(*res)
    except TimeoutError:
        ns_crash = True
        ns_multi_comp = False
        ns_verts = 0
    finally:
        signal.alarm(0)

    return {
        "pdb": name,
        "grid_scale": grid_scale,
        "ns_crash": ns_crash,
        "ns_multi_comp": ns_multi_comp,
        "ns_verts": ns_verts,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20, help="Number of workers")
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
        default="nanoshaper_discard_rate_vs_vertices.png",
        help="Output plot filename",
    )
    args = parser.parse_args()

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

    for gs in grid_scales:
        print(
            f"Processing Grid Scale: {gs} (PDBs: {len(pdb_files)}, Workers: {args.workers})"
        )
        task_args = [(p, gs) for p in pdb_files]
        n = len(task_args)
        gs_results = []

        with ctx.Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_one, task_args), 1):
                gs_results.append(r)
                if i % 100 == 0:
                    print(f"  {i}/{n} done", flush=True)

        results_by_scale[gs] = gs_results

    x_scales = []
    y_discarded = []
    y_vertices = []

    for gs in sorted(results_by_scale.keys()):
        rs = results_by_scale[gs]
        # Count discarded: has multiple comps >1% AND didn't crash
        discarded_count = sum(1 for r in rs if r["ns_multi_comp"] and not r["ns_crash"])
        discarded_pct = 100.0 * discarded_count / len(rs) if rs else 0

        valid_verts = [r["ns_verts"] for r in rs if not r["ns_crash"]]
        avg_verts = np.mean(valid_verts) if valid_verts else 0

        x_scales.append(gs)
        y_discarded.append(discarded_pct)
        y_vertices.append(avg_verts)

    print("\n--- Summary ---")
    print(f"{'Grid Scale':>10} | {'Discarded %':>11} | {'Avg Vertices':>12}")
    for gs, disc, verts in zip(x_scales, y_discarded, y_vertices):
        print(f"{gs:10.2f} | {disc:10.2f}% | {verts:12.1f}")

    summary_csv = FIG_DIR / "nanoshaper_discard_rate_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Grid Scale", "Discarded %", "Avg Vertices"])
        for gs, disc, verts in zip(x_scales, y_discarded, y_vertices):
            writer.writerow([gs, disc, verts])
    print(f"\nSummary data saved to {summary_csv}")

    raw_csv = FIG_DIR / "nanoshaper_discard_rate_raw.csv"
    if results_by_scale:
        all_raw = []
        for gs in sorted(results_by_scale.keys()):
            all_raw.extend(results_by_scale[gs])

        if all_raw:
            fields = list(all_raw[0].keys())
            with open(raw_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(all_raw)
            print(f"Raw data saved to {raw_csv}")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:red"
    ax1.set_xlabel("Grid Scale", fontsize=12)
    ax1.set_ylabel("Discarded Surfaces (% of total)", color=color1, fontsize=12)
    line1 = ax1.plot(
        x_scales, y_discarded, marker="o", color=color1, label="Discarded (%)"
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

    plt.title(
        "NanoShaper: Discarded Surfaces and Avg Vertices vs Grid Scale", fontsize=14
    )
    plt.tight_layout()

    out_path = FIG_DIR / args.output_img
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
