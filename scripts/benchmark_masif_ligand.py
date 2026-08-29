#!/usr/bin/env python3
"""
Benchmark: Surface Generation + Spectral Operator Time across methods (MasifLigand).

Methods: Alpha Complex (algo2, alpha=0), EDTSurf, MSMS, NanoShaper (grid scale sweep).
Measures surface generation time, spectral operator time, and total time per protein.
Wall clock per method batch for throughput.
"""

import argparse
import csv
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alphasurf.protein.create_operators import compute_operators  # noqa: E402
from alphasurf.protein.create_surface import mesh_simplification  # noqa: E402
from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402

PDB_DIR = os.path.join(
    project_root, "data", "masif_ligand", "raw_data_MasifLigand", "pdb"
)
NANOSHAPER_BIN = str(
    Path(project_root).parent / "nanoshaper-master" / "build" / "NanoShaper"
)
EDTSURF_BIN = str(Path(project_root).parent / "EDTSurf" / "EDTSurf")
CGAL_ALGO2_PATH = str(Path(project_root) / "cgal_alpha_bindings" / "build")
MSMS_PATH = os.path.join(project_root, "bin", "msms_linux", "msms")
PDB2XYZR_PATH = os.path.join(project_root, "bin", "msms_linux", "pdb_to_xyzr")

K_EIG = 128
MIN_VERTS_FOR_OPS = 128


def _set_thread_limits():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"


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


def _parse_pdb(pdb_path):
    parsed = parse_pdb_path(str(pdb_path), use_pqr=False)
    if parsed is None or parsed[5] is None or parsed[7] is None:
        return None
    return parsed[5], parsed[7]


def _surface_nanoshaper(atom_pos, atom_rad, grid_scale=0.5):
    work_dir = tempfile.mkdtemp(prefix="ns_bench_")
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
                f"Probe_Radius = 1.4\n"
                f"Max_Probes_Self_Intersections = 100\n"
                f"Self_Intersections_Grid_Coefficient = 1.5\n"
                f"Accurate_Triangulation = true\n"
                f"Triangulation = true\n"
                f"Check_duplicated_vertices = true\n"
                f"Save_Status_map = false\n"
                f"Save_PovRay = false\n"
            )
        t0 = time.time()
        result = subprocess.run(
            [NANOSHAPER_BIN, conf_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        t_gen = time.time() - t0
        if not os.path.exists(off_file):
            return None, t_gen, f"no_output(exit={result.returncode})"
        verts, faces = _parse_off(off_file)
        return (verts, faces), t_gen, None
    except subprocess.TimeoutExpired:
        return None, 120.0, "timeout"
    except Exception as e:
        return None, 0.0, str(e)[:80]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _surface_alpha_algo2(atom_pos, atom_rad, alpha_value=0.0):
    if CGAL_ALGO2_PATH not in sys.path:
        sys.path.insert(0, CGAL_ALGO2_PATH)
    import cgal_alpha_algo2

    t0 = time.time()
    pos = np.atleast_2d(atom_pos.astype(np.float32))
    rad = np.atleast_1d(atom_rad.astype(np.float32))
    result = cgal_alpha_algo2.compute_alpha_complex_algo2_from_atoms(
        pos, rad, alpha_value, 1.4
    )
    t_gen = time.time() - t0
    verts = np.asarray(result[0], dtype=np.float32)
    faces = np.asarray(result[1], dtype=np.int32)
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)
    return verts, faces, t_gen


def _surface_msms(pdb_path):
    work_dir = tempfile.mkdtemp(prefix="msms_bench_")
    xyzr_file = os.path.join(work_dir, "atoms.xyzr")
    out_name = os.path.join(work_dir, "msms_out")
    vert_file = out_name + ".vert"
    face_file = out_name + ".face"
    binary_dir = os.path.dirname(MSMS_PATH)

    try:
        with open(xyzr_file, "w") as f_out:
            subprocess.run(
                [PDB2XYZR_PATH, os.path.abspath(pdb_path)],
                stdout=f_out,
                stderr=subprocess.DEVNULL,
                cwd=binary_dir,
                timeout=60,
            )
        if not os.path.exists(xyzr_file) or os.path.getsize(xyzr_file) == 0:
            return None, 0.0, "xyzr_failed"

        density = 1.0
        n_verts = 0
        verts, faces = None, None
        t_gen_total = 0.0

        while n_verts < 256:
            t0 = time.time()
            result = subprocess.run(
                [
                    MSMS_PATH,
                    "-if",
                    xyzr_file,
                    "-of",
                    out_name,
                    "-density",
                    str(density),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=binary_dir,
                timeout=300,
            )
            t_gen_total += time.time() - t0

            if result.returncode != 0 or not os.path.exists(vert_file):
                return None, t_gen_total, f"msms_exit({result.returncode})"

            verts = np.loadtxt(
                vert_file, skiprows=3, dtype=np.float32, usecols=(0, 1, 2)
            )
            faces = (
                np.loadtxt(face_file, skiprows=3, dtype=np.int32, usecols=(0, 1, 2)) - 1
            )
            n_verts = len(verts)
            density += 1

        return (verts, faces), t_gen_total, None
    except subprocess.TimeoutExpired:
        return None, 300.0, "timeout"
    except Exception as e:
        return None, 0.0, str(e)[:80]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _surface_edtsurf(pdb_path, grid_scale=0.5):
    import trimesh

    out_base = os.path.join(tempfile.gettempdir(), f"edtsurf_{os.getpid()}")
    ply_file = out_base + ".ply"
    try:
        t0 = time.time()
        subprocess.run(
            [
                EDTSURF_BIN,
                "-i",
                pdb_path,
                "-o",
                out_base,
                "-s",
                "3",
                "-p",
                "1.4",
                "-f",
                str(grid_scale),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        t_gen = time.time() - t0
        if not os.path.exists(ply_file):
            return None, t_gen, "no_output"
        mesh = trimesh.load(ply_file, process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        return (verts, faces), t_gen, None
    except subprocess.TimeoutExpired:
        return None, 300.0, "timeout"
    except Exception as e:
        return None, 0.0, str(e)[:80]
    finally:
        for f in [ply_file, out_base + ".asa", out_base + "-cav.pdb"]:
            if os.path.exists(f):
                os.remove(f)


def _timeout_handler(signum, frame):
    raise TimeoutError("timed out")


def process_one(task):
    _set_thread_limits()
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(300)

    pdb_path, method, method_param = task
    name = Path(pdb_path).stem
    result = {
        "pdb": name,
        "method": method,
        "param": method_param,
        "crash": False,
        "valid": False,
        "n_verts": 0,
        "t_gen": 0.0,
        "t_simplify": 0.0,
        "t_spectral": 0.0,
        "t_total": 0.0,
    }

    try:
        parsed = _parse_pdb(pdb_path)
        if parsed is None:
            result["crash"] = True
            return result
        atom_pos, atom_rad = parsed

        if method == "nanoshaper":
            mesh, t_gen, err = _surface_nanoshaper(
                atom_pos, atom_rad, grid_scale=method_param
            )
            if err:
                result["crash"] = True
                result["t_gen"] = t_gen
                return result
        elif method == "alpha_algo2":
            verts, faces, t_gen = _surface_alpha_algo2(
                atom_pos, atom_rad, alpha_value=method_param
            )
            mesh = (verts, faces)
        elif method == "edtsurf":
            mesh, t_gen, err = _surface_edtsurf(pdb_path, grid_scale=method_param)
            if err:
                result["crash"] = True
                result["t_gen"] = t_gen
                return result
        elif method == "msms":
            mesh, t_gen, err = _surface_msms(pdb_path)
            if err:
                result["crash"] = True
                result["t_gen"] = t_gen
                return result
        else:
            result["crash"] = True
            return result

        verts, faces = mesh
        result["n_verts"] = len(verts)
        result["t_gen"] = t_gen

        # Cluster scan validation (same check as create_surface.py)
        if len(faces) > 0:
            import open3d as o3d

            o3d_mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces),
            )
            o3d_mesh.remove_degenerate_triangles()
            if method == "alpha_algo2":
                from alphasurf.protein.create_surface import (
                    cluster_triangles_by_vertex_sharing,
                )

                vc_clusters, vc_cluster_n = cluster_triangles_by_vertex_sharing(
                    np.asarray(o3d_mesh.triangles)
                )
                vc_cluster_n = np.asarray(vc_cluster_n)
                if len(vc_cluster_n) > 0:
                    largest = int(np.max(vc_cluster_n))
                    cutoff = int(0.01 * largest)
                    result["valid"] = int((vc_cluster_n >= cutoff).sum()) == 1
            else:
                tri_clusters, cluster_n, _ = o3d_mesh.cluster_connected_triangles()
                cluster_n = np.asarray(cluster_n)
                if len(cluster_n) > 0:
                    largest = int(np.max(cluster_n))
                    cutoff = int(0.01 * largest)
                    result["valid"] = int((cluster_n >= cutoff).sum()) == 1
        else:
            result["valid"] = False

        # MSMS surfaces need simplification (standard pipeline: 0.1 reduction)
        if method == "msms" and len(faces) > 0:
            t_simp = time.time()
            try:
                verts, faces, _, _ = mesh_simplification(
                    verts,
                    faces,
                    out_ply=None,
                    face_reduction_rate=0.1,
                    min_vert_number=16,
                    use_pymesh=False,
                    surface_method="msms",
                )
                result["t_simplify"] = time.time() - t_simp
                result["n_verts"] = len(verts)
            except Exception:
                result["t_simplify"] = -1.0
                result["crash"] = True
                return result

        if len(faces) > 0 and len(verts) >= MIN_VERTS_FOR_OPS:
            t0 = time.time()
            try:
                compute_operators(
                    verts,
                    faces,
                    k_eig=K_EIG,
                    use_fem_decomp=False,
                    use_robust_laplacian=False,
                )
                result["t_spectral"] = time.time() - t0
            except Exception:
                result["t_spectral"] = -1.0

        result["t_total"] = (
            result["t_gen"] + result["t_simplify"] + max(result["t_spectral"], 0.0)
        )

    except Exception:
        result["crash"] = True
    finally:
        signal.alarm(0)

    return result


def _save_csvs(all_results, wall_clock, csv_dir):
    raw_csv = os.path.join(csv_dir, "masif_benchmark_raw.csv")
    fields = list(all_results[0].keys())
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    # Compute total per method (including crashes) for validation rate
    total_per_method = defaultdict(int)
    for r in all_results:
        if not r["crash"]:
            total_per_method[(r["method"], r["param"])] += 1

    summary_csv = os.path.join(csv_dir, "masif_benchmark_summary.csv")
    groups = defaultdict(list)
    for r in all_results:
        if not r["crash"] and r["t_spectral"] > 0:
            groups[(r["method"], r["param"])].append(r)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Method",
                "Param",
                "N",
                "Avg Verts",
                "Avg Gen (s)",
                "Avg Simplify (s)",
                "Avg Spectral (s)",
                "Avg Total (s)",
                "Wall Clock (s)",
                "Throughput (s/protein)",
                "Valid Rate",
            ]
        )
        for key in sorted(groups.keys()):
            rs = groups[key]
            method, param = key
            avg_verts = np.mean([r["n_verts"] for r in rs])
            avg_gen = np.mean([r["t_gen"] for r in rs])
            avg_simp = np.mean([r["t_simplify"] for r in rs])
            avg_spec = np.mean([r["t_spectral"] for r in rs])
            avg_total = np.mean([r["t_total"] for r in rs])
            wc = wall_clock.get(key, 0.0)
            throughput = wc / len(rs) if len(rs) > 0 else 0.0
            n_valid = sum(1 for r in rs if r.get("valid", False))
            n_total = total_per_method.get(key, len(rs))
            valid_rate = n_valid / n_total if n_total > 0 else 0.0
            writer.writerow(
                [
                    method,
                    param,
                    len(rs),
                    f"{avg_verts:.1f}",
                    f"{avg_gen:.4f}",
                    f"{avg_simp:.4f}",
                    f"{avg_spec:.4f}",
                    f"{avg_total:.4f}",
                    f"{wc:.2f}",
                    f"{throughput:.4f}",
                    f"{valid_rate:.4f}",
                ]
            )

    print(f"  Saved {raw_csv} + {summary_csv}", flush=True)


def _run_batch(pool, tasks, method_label, workers):
    n = len(tasks)
    if n == 0:
        return [], 0.0
    print(f"\n  {method_label} ({n} tasks, {workers} workers)", flush=True)
    results = []
    t_wall_start = time.time()
    for i, r in enumerate(pool.imap_unordered(process_one, tasks), 1):
        results.append(r)
        if i % 50 == 0:
            print(f"    {i}/{n} done ({time.time() - t_wall_start:.1f}s)", flush=True)
    wall_clock = time.time() - t_wall_start
    print(f"    {n}/{n} done -- wall clock: {wall_clock:.1f}s", flush=True)
    return results, wall_clock


def plot_results(csv_dir, workers=0):
    import pandas as pd

    raw_csv = os.path.join(csv_dir, "masif_benchmark_raw.csv")
    df = pd.read_csv(raw_csv)
    df = df[
        (~df["crash"]) & (df["t_spectral"] > 0) & (df["n_verts"] >= MIN_VERTS_FOR_OPS)
    ]

    groups = df.groupby(["method", "param"])
    group_keys = sorted(groups.groups.keys())
    cmap = plt.cm.get_cmap("tab10", len(group_keys))
    color_map = {k: cmap(i) for i, k in enumerate(group_keys)}

    def _label(method, param):
        if method == "nanoshaper":
            return f"NanoShaper gs={param}"
        elif method == "alpha_algo2":
            return f"Alpha_algo2 (a={param})"
        else:
            return method

    # Unique methods for rows (group nanoshaper grid scales together)
    method_order = ["alpha_algo2", "edtsurf", "msms", "nanoshaper"]
    method_names = {
        "alpha_algo2": "Alpha Complex (algo2)",
        "edtsurf": "EDTSurf",
        "msms": "MSMS",
        "nanoshaper": "NanoShaper",
    }
    n_methods = len(method_order)
    fig, axes = plt.subplots(n_methods, 3, figsize=(21, 4 * n_methods), squeeze=False)

    col_info = [
        ("t_gen", "Surface Generation Time (s)", "Generation"),
        ("t_spectral", "Spectral Operator Time (s)", "Spectral Ops"),
        ("t_total", "Total Time (s)", "Total (gen+spectral)"),
    ]

    gs_colors = plt.cm.get_cmap("viridis", 6)

    for row_idx, base_method in enumerate(method_order):
        if base_method == "nanoshaper":
            sub_keys = [(m, p) for m, p in group_keys if m == "nanoshaper"]
        else:
            sub_keys = [(m, p) for m, p in group_keys if m == base_method]

        for col_idx, (y_col, y_label, col_title) in enumerate(col_info):
            ax = axes[row_idx, col_idx]

            for ki, (method, param) in enumerate(sub_keys):
                sub = groups.get_group((method, param))
                if len(sub) < 2:
                    continue
                if base_method == "nanoshaper":
                    c = gs_colors(ki)
                    lbl = f"gs={param}"
                else:
                    c = color_map[(method, param)]
                    lbl = _label(method, param)

                ax.scatter(
                    sub["n_verts"],
                    sub[y_col],
                    s=6,
                    alpha=0.3,
                    color=c,
                    label=lbl,
                    zorder=2,
                )

                # Trend line for large groups
                if len(sub) > 10:
                    xs = sub["n_verts"].values
                    n_bins = min(20, max(5, len(sub) // 20))
                    bins = pd.cut(xs, bins=n_bins)
                    grouped = (
                        sub.groupby(bins, observed=True)
                        .agg(
                            n_v_c=("n_verts", "mean"),
                            t_mean=(y_col, "mean"),
                            t_std=(y_col, "std"),
                        )
                        .dropna()
                    )
                    if len(grouped) > 1:
                        ax.plot(
                            grouped["n_v_c"],
                            grouped["t_mean"],
                            color=c,
                            linewidth=2,
                            zorder=4,
                        )
                        ax.fill_between(
                            grouped["n_v_c"],
                            grouped["t_mean"] - grouped["t_std"],
                            grouped["t_mean"] + grouped["t_std"],
                            color=c,
                            alpha=0.1,
                            zorder=1,
                        )

            ax.set_xlabel("Number of Vertices")
            ax.set_ylabel(y_label)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=7, markerscale=2)

            if row_idx == 0:
                ax.set_title(col_title)

        axes[row_idx, 0].annotate(
            method_names[base_method],
            xy=(0, 0.5),
            xytext=(-axes[row_idx, 0].yaxis.labelpad - 5, 0),
            xycoords=axes[row_idx, 0].yaxis.label,
            textcoords="offset points",
            size=14,
            ha="right",
            va="center",
            fontweight="bold",
        )

    fig.tight_layout()
    out_path = os.path.join(csv_dir, "masif_benchmark_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Distribution plot saved to {out_path}")

    # Bar chart: mean times per method
    summary_csv = os.path.join(csv_dir, "masif_benchmark_summary.csv")
    summary = pd.read_csv(summary_csv)
    if len(summary) == 0:
        return

    fig2, ax = plt.subplots(figsize=(max(14, len(summary) * 1.5), 6))
    labels = []
    for _, row in summary.iterrows():
        m, p = row["Method"], row["Param"]
        if m == "nanoshaper":
            labels.append(f"NanoShaper\ngs={p}")
        elif m == "alpha_algo2":
            labels.append(f"Alpha_algo2\n(a={p})")
        else:
            labels.append(m.capitalize())

    x = np.arange(len(labels))
    width = 0.25
    ax.bar(
        x - width, summary["Avg Gen (s)"], width, label="Generation", color="tab:blue"
    )
    ax.bar(
        x, summary["Avg Spectral (s)"], width, label="Spectral Ops", color="tab:orange"
    )
    ax.bar(
        x + width,
        summary["Throughput (s/protein)"],
        width,
        label="Throughput (wall/N)",
        color="tab:green",
        alpha=0.7,
    )

    for i, row in summary.iterrows():
        ax.text(
            i,
            0,
            f"{row['Avg Verts']:.0f}v",
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
        )

    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.title(f"MasifLigand Benchmark: Surface Gen + Spectral Ops ({workers} workers)")
    fig2.tight_layout()
    out_path2 = os.path.join(csv_dir, "masif_benchmark_bar.png")
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Bar chart saved to {out_path2}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark surface methods on MasifLigand"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pdb-dir", type=str, default=PDB_DIR)
    parser.add_argument(
        "--grid-scales",
        type=str,
        default="0.3,0.4,0.5,0.6,0.8",
        help="Comma-separated NanoShaper grid scales",
    )
    parser.add_argument("--alpha-value", type=float, default=0.0)
    parser.add_argument("--edtsurf-grid-scale", type=float, default=0.5)
    parser.add_argument(
        "--methods",
        type=str,
        default="alpha_algo2,edtsurf,msms,nanoshaper",
        help="Comma-separated methods to run",
    )
    parser.add_argument(
        "--plot-only", action="store_true", help="Skip benchmark, just plot from CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for benchmark CSVs and plots (default: script directory)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.abspath(args.output_dir or script_dir)
    os.makedirs(csv_dir, exist_ok=True)
    os.chdir(script_dir)

    if args.plot_only:
        plot_results(csv_dir, args.workers)
        return

    methods_to_run = set(m.strip() for m in args.methods.split(","))
    grid_scales = [float(x.strip()) for x in args.grid_scales.split(",")]

    pdb_files = sorted(str(p) for p in Path(args.pdb_dir).glob("*.pdb"))
    if args.max_files:
        pdb_files = pdb_files[: args.max_files]
    if not pdb_files:
        parser.error(f"no PDB files found in {args.pdb_dir}")

    print(f"PDBs: {len(pdb_files)}")
    print(f"Methods: {', '.join(sorted(methods_to_run))} (gs={grid_scales})")
    print(f"Workers: {args.workers}")

    all_results = []
    wall_clock = {}
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(args.workers) as pool:
        if "alpha_algo2" in methods_to_run:
            tasks = [(p, "alpha_algo2", args.alpha_value) for p in pdb_files]
            rs, wc = _run_batch(
                pool, tasks, f"Alpha_algo2 (a={args.alpha_value})", args.workers
            )
            all_results.extend(rs)
            wall_clock[("alpha_algo2", args.alpha_value)] = wc
            _save_csvs(all_results, wall_clock, csv_dir)

        if "edtsurf" in methods_to_run:
            tasks = [(p, "edtsurf", args.edtsurf_grid_scale) for p in pdb_files]
            rs, wc = _run_batch(
                pool,
                tasks,
                f"EDTSurf (gs={args.edtsurf_grid_scale})",
                args.workers,
            )
            all_results.extend(rs)
            wall_clock[("edtsurf", args.edtsurf_grid_scale)] = wc
            _save_csvs(all_results, wall_clock, csv_dir)

        if "msms" in methods_to_run:
            tasks = [(p, "msms", 0) for p in pdb_files]
            rs, wc = _run_batch(pool, tasks, "MSMS", args.workers)
            all_results.extend(rs)
            wall_clock[("msms", 0)] = wc
            _save_csvs(all_results, wall_clock, csv_dir)

        if "nanoshaper" in methods_to_run:
            for gs in grid_scales:
                tasks = [(p, "nanoshaper", gs) for p in pdb_files]
                rs, wc = _run_batch(pool, tasks, f"NanoShaper (gs={gs})", args.workers)
                all_results.extend(rs)
                wall_clock[("nanoshaper", gs)] = wc
                _save_csvs(all_results, wall_clock, csv_dir)

    # Print summary
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["method"], r["param"])].append(r)

    print("\n" + "=" * 142)
    print(
        f"{'Method':>16} | {'Param':>6} | {'N':>5} | {'Avg Verts':>10} | "
        f"{'Avg Gen(s)':>10} | {'Avg Simp(s)':>11} | {'Avg Spect(s)':>12} | {'Avg Total(s)':>12} | "
        f"{'Wall Clock':>11} | {'Throughput':>12} | {'Valid Rate':>10}"
    )
    print("-" * 142)
    for key in sorted(groups.keys()):
        method, param = key
        valid = [r for r in groups[key] if not r["crash"] and r["t_spectral"] > 0]
        if not valid:
            continue
        avg_verts = np.mean([r["n_verts"] for r in valid])
        avg_gen = np.mean([r["t_gen"] for r in valid])
        avg_simp = np.mean([r["t_simplify"] for r in valid])
        avg_spec = np.mean([r["t_spectral"] for r in valid])
        avg_total = np.mean([r["t_total"] for r in valid])
        wc = wall_clock.get(key, 0.0)
        throughput = wc / len(valid) if len(valid) > 0 else 0.0
        n_valid = sum(1 for r in valid if r.get("valid", False))
        n_total_noncrash = sum(1 for r in groups[key] if not r["crash"])
        vr = n_valid / n_total_noncrash if n_total_noncrash > 0 else 0.0
        print(
            f"{method:>16} | {param:>6.2f} | {len(valid):>5} | {avg_verts:10.1f} | "
            f"{avg_gen:10.4f} | {avg_simp:11.4f} | {avg_spec:12.4f} | {avg_total:12.4f} | "
            f"{wc:11.1f} | {throughput:12.4f} | {vr:10.4f}"
        )
    print("=" * 142)

    total_wall = sum(wall_clock.values())
    print(f"\nGrand total wall clock: {total_wall:.1f}s ({total_wall / 60:.1f}min)")
    print(f"Total proteins: {len(pdb_files)}")
    print(f"Time per protein (all methods): {total_wall / len(pdb_files):.4f}s")

    plot_results(csv_dir, args.workers)


if __name__ == "__main__":
    main()
