#!/usr/bin/env python3
"""
Benchmark: Surface Generation + Spectral Operator Time across methods (PINDER).

Methods: Alpha Complex (alpha=0), EDTSurf, NanoShaper (grid scale sweep).
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
from contextlib import nullcontext
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

PDB_DIR = os.path.join(project_root, "data", "pinder-pair", "pdb")
NANOSHAPER_BIN = str(
    Path(project_root).parent / "nanoshaper-master" / "build" / "NanoShaper"
)
EDTSURF_BIN = str(Path(project_root).parent / "EDTSurf" / "EDTSurf")
CGAL_PATH = str(Path(project_root) / "cgal_alpha_bindings" / "build_py310")
CGAL_ALGO2_PATH = str(Path(project_root) / "cgal_alpha_bindings" / "build")
MSMS_PATH = os.path.join(project_root, "bin", "msms_linux", "msms")

K_EIG = 128
MIN_VERTS_FOR_OPS = 128


def _set_thread_limits():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_MAX_THREADS"] = "1"


def select_pdb_files(sample_lr, seed):
    pdb_paths = list(Path(PDB_DIR).glob("*.pdb"))
    if sample_lr is None:
        holo_stems = {
            p.stem.replace("_holo", "") for p in pdb_paths if p.stem.endswith("_holo")
        }
        return sorted(str(p) for p in pdb_paths if p.stem not in holo_stems)

    left = sorted(p for p in pdb_paths if p.stem.endswith("_L"))
    right = sorted(p for p in pdb_paths if p.stem.endswith("_R"))
    if len(left) < sample_lr or len(right) < sample_lr:
        raise ValueError(
            f"Requested {sample_lr} PDBs per side, found {len(left)} L and {len(right)} R"
        )

    rng = np.random.default_rng(seed)
    selected_left = rng.choice(left, size=sample_lr, replace=False)
    selected_right = rng.choice(right, size=sample_lr, replace=False)
    return sorted(str(p) for p in (*selected_left, *selected_right))


def write_sample_manifest(pdb_files, output_dir, seed):
    manifest_path = Path(output_dir) / "pdb_sample_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pdb", "side", "seed"])
        for pdb_file in pdb_files:
            side = Path(pdb_file).stem.rsplit("_", 1)[-1]
            writer.writerow([Path(pdb_file).name, side, seed])
    return manifest_path


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


def _new_surface_timings():
    return {"t_prepare": 0.0, "t_engine": 0.0, "t_output_parse": 0.0}


def _surface_nanoshaper(atom_pos, atom_rad, grid_scale=0.5):
    timings = _new_surface_timings()
    prepare_start = time.perf_counter()
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
        timings["t_prepare"] = time.perf_counter() - prepare_start
        engine_start = time.perf_counter()
        result = subprocess.run(
            [NANOSHAPER_BIN, conf_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        timings["t_engine"] = time.perf_counter() - engine_start
        if not os.path.exists(off_file):
            return None, timings, f"no_output(exit={result.returncode})"
        parse_start = time.perf_counter()
        verts, faces = _parse_off(off_file)
        timings["t_output_parse"] = time.perf_counter() - parse_start
        return (verts, faces), timings, None
    except subprocess.TimeoutExpired:
        timings["t_engine"] = time.perf_counter() - engine_start
        return None, timings, "timeout"
    except Exception as e:
        return None, timings, str(e)[:80]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _surface_alpha(atom_pos, atom_rad):
    timings = _new_surface_timings()
    prepare_start = time.perf_counter()
    if CGAL_PATH not in sys.path:
        sys.path.insert(0, CGAL_PATH)
    import cgal_alpha

    timings["t_prepare"] = time.perf_counter() - prepare_start
    engine_start = time.perf_counter()
    verts, faces = cgal_alpha.compute_alpha_complex_from_atoms(
        atom_pos, atom_rad, 0.0, 1.4, "singular+regular"
    )
    timings["t_engine"] = time.perf_counter() - engine_start
    parse_start = time.perf_counter()
    mesh = np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)
    timings["t_output_parse"] = time.perf_counter() - parse_start
    return mesh, timings, None


def _surface_alpha_algo2(atom_pos, atom_rad):
    timings = _new_surface_timings()
    prepare_start = time.perf_counter()
    if CGAL_ALGO2_PATH not in sys.path:
        sys.path.insert(0, CGAL_ALGO2_PATH)
    import cgal_alpha_algo2

    pos = np.atleast_2d(atom_pos.astype(np.float32))
    rad = np.atleast_1d(atom_rad.astype(np.float32))
    timings["t_prepare"] = time.perf_counter() - prepare_start
    engine_start = time.perf_counter()
    result = cgal_alpha_algo2.compute_alpha_complex_algo2_from_atoms(pos, rad, 0.0, 1.4)
    timings["t_engine"] = time.perf_counter() - engine_start
    parse_start = time.perf_counter()
    verts = np.asarray(result[0], dtype=np.float32)
    faces = np.asarray(result[1], dtype=np.int32)
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)
    if faces.ndim == 1:
        faces = faces.reshape(-1, 3)
    timings["t_output_parse"] = time.perf_counter() - parse_start
    return (verts, faces), timings, None


def _surface_msms(atom_pos, atom_rad):
    timings = _new_surface_timings()
    prepare_start = time.perf_counter()
    work_dir = tempfile.mkdtemp(prefix="msms_bench_")
    xyzr_file = os.path.join(work_dir, "atoms.xyzr")
    out_name = os.path.join(work_dir, "msms_out")
    vert_file = out_name + ".vert"
    face_file = out_name + ".face"
    binary_dir = os.path.dirname(MSMS_PATH)
    engine_start = None

    try:
        with open(xyzr_file, "w") as f:
            for pos, radius in zip(atom_pos, atom_rad):
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {float(radius):.6f}\n")
        timings["t_prepare"] = time.perf_counter() - prepare_start

        density = 1.0
        n_verts = 0
        verts, faces = None, None
        while n_verts < 256:
            engine_start = time.perf_counter()
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
            timings["t_engine"] += time.perf_counter() - engine_start
            engine_start = None

            if result.returncode != 0 or not os.path.exists(vert_file):
                return None, timings, f"msms_exit({result.returncode})"

            parse_start = time.perf_counter()
            verts = np.loadtxt(
                vert_file, skiprows=3, dtype=np.float32, usecols=(0, 1, 2)
            )
            faces = (
                np.loadtxt(face_file, skiprows=3, dtype=np.int32, usecols=(0, 1, 2)) - 1
            )
            timings["t_output_parse"] += time.perf_counter() - parse_start
            n_verts = len(verts)
            density += 1

        return (verts, faces), timings, None
    except subprocess.TimeoutExpired:
        if engine_start is None:
            timings["t_prepare"] = time.perf_counter() - prepare_start
        else:
            timings["t_engine"] += time.perf_counter() - engine_start
        return None, timings, "timeout"
    except Exception as e:
        return None, timings, str(e)[:80]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _surface_edtsurf(pdb_path, grid_scale=0.5):
    import trimesh

    timings = _new_surface_timings()
    prepare_start = time.perf_counter()
    out_base = os.path.join(tempfile.gettempdir(), f"edtsurf_{os.getpid()}")
    ply_file = out_base + ".ply"
    timings["t_prepare"] = time.perf_counter() - prepare_start
    try:
        engine_start = time.perf_counter()
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
        timings["t_engine"] = time.perf_counter() - engine_start
        if not os.path.exists(ply_file):
            return None, timings, "no_output"
        parse_start = time.perf_counter()
        mesh = trimesh.load(ply_file, process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        timings["t_output_parse"] = time.perf_counter() - parse_start
        return (verts, faces), timings, None
    except subprocess.TimeoutExpired:
        timings["t_engine"] = time.perf_counter() - engine_start
        return None, timings, "timeout"
    except Exception as e:
        return None, timings, str(e)[:80]
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
    process_start = time.perf_counter()

    pdb_path, method, method_param, surface_only = task
    name = Path(pdb_path).stem
    result = {
        "pdb": name,
        "method": method,
        "param": method_param,
        "crash": False,
        "n_verts": 0,
        "n_verts_raw": 0,
        "t_input": 0.0,
        "t_prepare": 0.0,
        "t_engine": 0.0,
        "t_output_parse": 0.0,
        "t_gen": 0.0,
        "t_simplify": 0.0,
        "t_spectral": 0.0,
        "t_total": 0.0,
        "t_end_to_end": 0.0,
    }

    try:
        input_start = time.perf_counter()
        parsed = _parse_pdb(pdb_path)
        result["t_input"] = time.perf_counter() - input_start
        if parsed is None:
            result["crash"] = True
            return result
        atom_pos, atom_rad = parsed

        if method == "nanoshaper":
            mesh, method_timings, err = _surface_nanoshaper(
                atom_pos, atom_rad, grid_scale=method_param
            )
        elif method == "alpha":
            mesh, method_timings, err = _surface_alpha(atom_pos, atom_rad)
        elif method == "alpha_algo2":
            mesh, method_timings, err = _surface_alpha_algo2(atom_pos, atom_rad)
        elif method == "edtsurf":
            mesh, method_timings, err = _surface_edtsurf(
                pdb_path, grid_scale=method_param
            )
        elif method == "msms":
            mesh, method_timings, err = _surface_msms(atom_pos, atom_rad)
        else:
            result["crash"] = True
            return result

        result.update(method_timings)
        result["t_gen"] = result["t_engine"]
        if err:
            result["crash"] = True
            return result

        verts, faces = mesh
        result["n_verts_raw"] = len(verts)
        result["n_verts"] = len(verts)
        if method == "msms":
            simplify_start = time.perf_counter()
            verts, faces, _, _ = mesh_simplification(
                verts,
                faces,
                out_ply=None,
                face_reduction_rate=method_param,
                min_vert_number=16,
                max_vert_number=100000,
                use_pymesh=False,
                surface_method="msms",
                allow_multiple_components=True,
            )
            result["t_simplify"] = time.perf_counter() - simplify_start
            result["n_verts"] = len(verts)
        if surface_only:
            result["t_total"] = result["t_engine"] + result["t_simplify"]
            return result

        if len(faces) > 0 and len(verts) >= MIN_VERTS_FOR_OPS:
            spectral_start = time.perf_counter()
            try:
                compute_operators(
                    verts,
                    faces,
                    k_eig=K_EIG,
                    use_fem_decomp=False,
                    use_robust_laplacian=False,
                )
                result["t_spectral"] = time.perf_counter() - spectral_start
            except Exception:
                result["t_spectral"] = -1.0

        result["t_total"] = (
            result["t_gen"] + result["t_simplify"] + max(result["t_spectral"], 0.0)
        )

    except Exception:
        result["crash"] = True
    finally:
        result["t_end_to_end"] = time.perf_counter() - process_start
        signal.alarm(0)

    return result


def _save_csvs(all_results, wall_clock, csv_dir):
    raw_csv = os.path.join(csv_dir, "pinder_benchmark_raw.csv")
    fields = list(all_results[0].keys())
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_results)

    summary_csv = os.path.join(csv_dir, "pinder_benchmark_summary.csv")
    groups = defaultdict(list)
    for r in all_results:
        if not r["crash"] and r["t_gen"] > 0:
            groups[(r["method"], r["param"])].append(r)

    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Method",
                "Param",
                "N",
                "Avg Verts",
                "Avg Input (s)",
                "Avg Prepare (s)",
                "Avg Engine (s)",
                "Avg Output Parse (s)",
                "Avg Simplify (s)",
                "Avg End-to-End (s)",
                "Avg Spectral (s)",
                "Avg Total (s)",
                "Wall Clock (s)",
                "Throughput (s/protein)",
            ]
        )
        for key in sorted(groups.keys()):
            rs = groups[key]
            method, param = key
            avg_verts = np.mean([r["n_verts"] for r in rs])
            spectral = [r["t_spectral"] for r in rs if r["t_spectral"] > 0]
            avg_spec = np.mean(spectral) if spectral else 0.0
            avg_total = np.mean([r["t_total"] for r in rs])
            wc = wall_clock.get(key, 0.0)
            throughput = wc / len(rs) if len(rs) > 0 else 0.0
            writer.writerow(
                [
                    method,
                    param,
                    len(rs),
                    f"{avg_verts:.1f}",
                    f"{np.mean([r['t_input'] for r in rs]):.4f}",
                    f"{np.mean([r['t_prepare'] for r in rs]):.4f}",
                    f"{np.mean([r['t_engine'] for r in rs]):.4f}",
                    f"{np.mean([r['t_output_parse'] for r in rs]):.4f}",
                    f"{np.mean([r['t_simplify'] for r in rs]):.4f}",
                    f"{np.mean([r['t_end_to_end'] for r in rs]):.4f}",
                    f"{avg_spec:.4f}",
                    f"{avg_total:.4f}",
                    f"{wc:.2f}",
                    f"{throughput:.4f}",
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
    iterator = (
        map(process_one, tasks)
        if pool is None
        else pool.imap_unordered(process_one, tasks)
    )
    for i, r in enumerate(iterator, 1):
        results.append(r)
        if i % 100 == 0:
            print(f"    {i}/{n} done ({time.time() - t_wall_start:.1f}s)", flush=True)
    wall_clock = time.time() - t_wall_start
    print(f"    {n}/{n} done -- wall clock: {wall_clock:.1f}s", flush=True)
    return results, wall_clock


def plot_results(csv_dir, workers=0):
    import pandas as pd
    from scipy.stats import gaussian_kde

    raw_csv = os.path.join(csv_dir, "pinder_benchmark_raw.csv")
    df = pd.read_csv(raw_csv)
    df = df[
        (~df["crash"]) & (df["t_spectral"] > 0) & (df["n_verts"] >= MIN_VERTS_FOR_OPS)
    ]

    methods = df["method"].unique()
    cmap = plt.cm.get_cmap("tab10", len(methods))
    color_map = {m: cmap(i) for i, m in enumerate(sorted(methods))}

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Plot 1: n_verts vs t_spectral (scatter + KDE)
    ax1 = axes[0]
    for method in sorted(methods):
        sub = df[df["method"] == method]
        if len(sub) < 2:
            continue
        c = color_map[method]
        ax1.scatter(
            sub["n_verts"],
            sub["t_spectral"],
            s=6,
            alpha=0.3,
            color=c,
            label=method,
            zorder=2,
        )

        if len(sub) > 10:
            xs = sub["n_verts"].values
            ys = sub["t_spectral"].values
            try:
                kde = gaussian_kde(np.vstack([xs, ys]))
                x_grid = np.linspace(xs.min(), xs.max(), 200)
                y_grid = np.linspace(ys.min(), ys.max(), 200)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = np.reshape(kde(np.vstack([X.ravel(), Y.ravel()])), X.shape)
                ax1.contour(
                    X, Y, Z, levels=5, colors=[c], alpha=0.5, linewidths=1.0, zorder=3
                )
            except Exception:
                pass

            n_bins = min(30, max(5, len(sub) // 20))
            bins = pd.cut(xs, bins=n_bins)
            grouped = (
                sub.groupby(bins, observed=True)
                .agg(
                    n_v_c=("n_verts", "mean"),
                    t_mean=("t_spectral", "mean"),
                    t_std=("t_spectral", "std"),
                )
                .dropna()
            )
            if len(grouped) > 1:
                ax1.plot(
                    grouped["n_v_c"], grouped["t_mean"], color=c, linewidth=2, zorder=4
                )
                ax1.fill_between(
                    grouped["n_v_c"],
                    grouped["t_mean"] - grouped["t_std"],
                    grouped["t_mean"] + grouped["t_std"],
                    color=c,
                    alpha=0.1,
                    zorder=1,
                )

    ax1.set_xlabel("Number of Vertices")
    ax1.set_ylabel("Spectral Operator Time (s)")
    ax1.set_title("Spectral Operator Time Distribution")
    ax1.legend(fontsize=8, markerscale=3)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.25)

    # Plot 2: n_verts vs t_gen
    ax2 = axes[1]
    for method in sorted(methods):
        sub = df[df["method"] == method]
        if len(sub) < 2:
            continue
        c = color_map[method]
        ax2.scatter(
            sub["n_verts"],
            sub["t_gen"],
            s=6,
            alpha=0.3,
            color=c,
            label=method,
            zorder=2,
        )

    ax2.set_xlabel("Number of Vertices")
    ax2.set_ylabel("Surface Generation Time (s)")
    ax2.set_title("Surface Generation Time Distribution")
    ax2.legend(fontsize=8, markerscale=3)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.25)

    # Plot 3: n_verts vs t_total
    ax3 = axes[2]
    for method in sorted(methods):
        sub = df[df["method"] == method]
        if len(sub) < 2:
            continue
        c = color_map[method]
        ax3.scatter(
            sub["n_verts"],
            sub["t_total"],
            s=6,
            alpha=0.3,
            color=c,
            label=method,
            zorder=2,
        )

    ax3.set_xlabel("Number of Vertices")
    ax3.set_ylabel("Total Time (s)")
    ax3.set_title("Total Time (gen + spectral) Distribution")
    ax3.legend(fontsize=8, markerscale=3)
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = os.path.join(csv_dir, "pinder_benchmark_distribution.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Distribution plot saved to {out_path}")

    # Bar chart: mean times per method
    summary_csv = os.path.join(csv_dir, "pinder_benchmark_summary.csv")
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
        elif m == "alpha":
            labels.append(f"Alpha\n(a={p})")
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
    plt.title(f"PINDER Benchmark: Surface Gen + Spectral Ops ({workers} workers)")
    fig2.tight_layout()
    out_path2 = os.path.join(csv_dir, "pinder_benchmark_bar.png")
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Bar chart saved to {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark surface methods on PINDER")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--sample-lr",
        type=int,
        default=None,
        help="Randomly select this many L and this many R PDBs",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--methods",
        default="alpha_algo2,edtsurf,msms,nanoshaper",
        help="Comma-separated methods to benchmark",
    )
    parser.add_argument("--edtsurf-grid-scale", type=float, default=0.5)
    parser.add_argument("--msms-reduction-rate", type=float, default=0.1)
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help="Measure PDB parsing and surface generation without spectral operators",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for raw/summary CSVs and plots",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory containing PDB files (defaults to the PINDER directory)",
    )
    parser.add_argument(
        "--grid-scales",
        type=str,
        default="0.3,0.4,0.5,0.6,0.8",
        help="Comma-separated NanoShaper grid scales",
    )
    parser.add_argument("--alpha-value", type=float, default=0.0)
    parser.add_argument(
        "--plot-only", action="store_true", help="Skip benchmark, just plot from CSV"
    )
    args = parser.parse_args()

    csv_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(csv_dir)

    if args.plot_only:
        plot_results(csv_dir, args.workers)
        return

    methods = {method.strip() for method in args.methods.split(",") if method.strip()}
    known_methods = {"alpha_algo2", "edtsurf", "msms", "nanoshaper"}
    unknown_methods = methods - known_methods
    if unknown_methods:
        raise SystemExit(f"Unknown methods: {', '.join(sorted(unknown_methods))}")
    if args.max_files is not None and args.sample_lr is not None:
        raise SystemExit("Use either --max-files or --sample-lr, not both")

    global PDB_DIR
    if args.pdb_dir is not None:
        PDB_DIR = str(args.pdb_dir.resolve())

    grid_scales = [float(x.strip()) for x in args.grid_scales.split(",")]
    csv_dir = args.output_dir or Path(os.path.dirname(os.path.abspath(__file__)))
    csv_dir = Path(csv_dir).resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = select_pdb_files(args.sample_lr, args.seed)
    if args.max_files:
        pdb_files = pdb_files[: args.max_files]
    if args.sample_lr is not None:
        manifest_path = write_sample_manifest(pdb_files, csv_dir, args.seed)
        print(f"Sample manifest: {manifest_path}")

    print(f"PDBs: {len(pdb_files)}")
    print(f"Methods: {sorted(methods)}")
    print(f"EDTSurf grid scale: {args.edtsurf_grid_scale}")
    print(f"MSMS face reduction rate: {args.msms_reduction_rate}")
    print(f"NanoShaper grid scales: {grid_scales}")
    print(f"Workers: {args.workers}")

    all_results = []
    wall_clock = {}
    ctx = multiprocessing.get_context("spawn")

    pool_context = ctx.Pool(args.workers) if args.workers > 1 else nullcontext(None)
    with pool_context as pool:
        if "alpha_algo2" in methods:
            tasks = [
                (p, "alpha_algo2", args.alpha_value, args.surface_only)
                for p in pdb_files
            ]
            rs, wc = _run_batch(
                pool, tasks, f"Alpha_algo2 (a={args.alpha_value})", args.workers
            )
            all_results.extend(rs)
            wall_clock[("alpha_algo2", args.alpha_value)] = wc
            _save_csvs(all_results, wall_clock, csv_dir)

        if "edtsurf" in methods:
            tasks = [
                (p, "edtsurf", args.edtsurf_grid_scale, args.surface_only)
                for p in pdb_files
            ]
            rs, wc = _run_batch(
                pool,
                tasks,
                f"EDTSurf (gs={args.edtsurf_grid_scale})",
                args.workers,
            )
            all_results.extend(rs)
            wall_clock[("edtsurf", args.edtsurf_grid_scale)] = wc
            _save_csvs(all_results, wall_clock, csv_dir)

        if "msms" in methods:
            tasks = [
                (p, "msms", args.msms_reduction_rate, args.surface_only)
                for p in pdb_files
            ]
            rs, wc = _run_batch(pool, tasks, "MSMS", args.workers)
            all_results.extend(rs)
            wall_clock[("msms", args.msms_reduction_rate)] = wc
            _save_csvs(all_results, wall_clock, csv_dir)

        if "nanoshaper" in methods:
            for gs in grid_scales:
                tasks = [(p, "nanoshaper", gs, args.surface_only) for p in pdb_files]
                rs, wc = _run_batch(
                    pool,
                    tasks,
                    f"NanoShaper (gs={gs})",
                    args.workers,
                )
                all_results.extend(rs)
                wall_clock[("nanoshaper", gs)] = wc
                _save_csvs(all_results, wall_clock, csv_dir)

    # Print summary
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["method"], r["param"])].append(r)

    print("\n" + "=" * 158)
    print(
        f"{'Method':>16} | {'Param':>6} | {'N':>5} | {'Avg Verts':>10} | "
        f"{'Input(s)':>9} | {'Prep(s)':>9} | {'Engine(s)':>10} | "
        f"{'Parse(s)':>9} | {'Simplify(s)':>11} | {'End2End(s)':>10} | "
        f"{'Wall Clock':>11} | {'Throughput':>12}"
    )
    print("-" * 158)
    for key in sorted(groups.keys()):
        method, param = key
        valid = [r for r in groups[key] if not r["crash"] and r["t_gen"] > 0]
        if not valid:
            continue
        avg_verts = np.mean([r["n_verts"] for r in valid])
        wc = wall_clock.get(key, 0.0)
        throughput = wc / len(valid) if len(valid) > 0 else 0.0
        print(
            f"{method:>16} | {param:>6.2f} | {len(valid):>5} | {avg_verts:10.1f} | "
            f"{np.mean([r['t_input'] for r in valid]):9.4f} | "
            f"{np.mean([r['t_prepare'] for r in valid]):9.4f} | "
            f"{np.mean([r['t_engine'] for r in valid]):10.4f} | "
            f"{np.mean([r['t_output_parse'] for r in valid]):9.4f} | "
            f"{np.mean([r['t_simplify'] for r in valid]):11.4f} | "
            f"{np.mean([r['t_end_to_end'] for r in valid]):10.4f} | "
            f"{wc:11.1f} | {throughput:12.4f}"
        )
    print("=" * 158)

    total_wall = sum(wall_clock.values())
    print(f"\nGrand total wall clock: {total_wall:.1f}s ({total_wall / 60:.1f}min)")
    print(f"Total proteins: {len(pdb_files)}")
    print(f"Time per protein (all methods): {total_wall / len(pdb_files):.4f}s")

    if not args.surface_only:
        plot_results(csv_dir, args.workers)


if __name__ == "__main__":
    main()
