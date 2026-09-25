#!/usr/bin/env python3
"""
Plot spectral operator time distribution per method.

Each point = one protein system.
X = number of surface vertices, Y = spectral operator time.
Multiple series: NanoShaper (various grid scales), EDTSurf, Alpha Complex.
Each series gets its own color with a KDE / confidence band overlay.
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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = str(Path(__file__).resolve().parents[2])
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "benchmarks"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alphasurf.protein.create_operators import compute_operators
from alphasurf.protein.graphs import parse_pdb_path

PDB_DIR = os.path.join(project_root, "data", "pinder-pair", "pdb")

NANOSHAPER_BIN = str(
    Path(project_root).parent / "nanoshaper-master" / "build" / "NanoShaper"
)
EDTSURF_BIN = str(Path(project_root).parent / "EDTSurf" / "EDTSurf")

K_EIG = 128
MIN_VERTS_FOR_OPS = 128


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
        return None, "parse_failed"
    atom_pos, atom_rad = parsed[5], parsed[7]
    return (atom_pos, atom_rad), None


def _surface_nanoshaper(atom_pos, atom_rad, grid_scale=0.5):
    work_dir = tempfile.mkdtemp(prefix="ns_dist_")
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
        result = subprocess.run(
            [NANOSHAPER_BIN, conf_file],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not os.path.exists(off_file):
            raise RuntimeError(f"NanoShaper no output (exit={result.returncode})")
        return _parse_off(off_file)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _surface_alpha(atom_pos, atom_rad):
    cgal_path = str(Path(project_root) / "cgal_alpha_bindings" / "build_py310")
    if cgal_path not in sys.path:
        sys.path.insert(0, cgal_path)
    import cgal_alpha

    verts, faces = cgal_alpha.compute_alpha_complex_from_atoms(
        atom_pos, atom_rad, 0.0, 1.4, "singular+regular"
    )
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _surface_edtsurf(pdb_path):
    import trimesh

    out_base = os.path.join(tempfile.gettempdir(), f"edtsurf_{os.getpid()}")
    ply_file = out_base + ".ply"
    try:
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
                "0.5",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if not os.path.exists(ply_file):
            return None, None
        mesh = trimesh.load(ply_file, process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        return verts, faces
    finally:
        for f in [ply_file, out_base + ".asa", out_base + "-cav.pdb"]:
            if os.path.exists(f):
                os.remove(f)


def _time_operators(verts, faces):
    if len(verts) < MIN_VERTS_FOR_OPS or len(faces) == 0:
        return 0.0, len(verts)
    t0 = time.time()
    try:
        compute_operators(
            verts, faces, k_eig=K_EIG, use_fem_decomp=False, use_robust_laplacian=False
        )
        return time.time() - t0, len(verts)
    except Exception:
        return -1.0, len(verts)


def _timeout_handler(signum, frame):
    raise TimeoutError("timed out")


def process_one(args):
    pdb_path, method = args
    name = Path(pdb_path).stem

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(300)

    row = {
        "pdb": name,
        "method": method,
        "n_verts": 0,
        "operator_time": 0.0,
        "error": "",
    }

    try:
        parsed, err = _parse_pdb(pdb_path)
        if err:
            row["error"] = "parse_failed"
            row["operator_time"] = -1.0
            return row

        atom_pos, atom_rad = parsed

        try:
            if method.startswith("nanoshaper_gs"):
                gs = float(method.split("gs")[1])
                verts, faces = _surface_nanoshaper(atom_pos, atom_rad, grid_scale=gs)
            elif method == "alpha":
                verts, faces = _surface_alpha(atom_pos, atom_rad)
            elif method == "edtsurf":
                verts, faces = _surface_edtsurf(pdb_path)
                if verts is None:
                    row["error"] = "edtsurf_no_output"
                    return row
            else:
                row["error"] = f"unknown_method:{method}"
                return row

            op_time, n_v = _time_operators(verts, faces)
            row["n_verts"] = n_v
            row["operator_time"] = op_time
            if op_time < 0:
                row["error"] = "operator_failed"

        except TimeoutError:
            row["error"] = "timeout"
        except Exception as e:
            row["error"] = str(e)[:60]

    except TimeoutError:
        row["error"] = "timeout"
    finally:
        signal.alarm(0)

    return row


def save_csv(all_rows, csv_path):
    fields = ["pdb", "method", "n_verts", "operator_time", "error"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


def plot_distribution(csv_path, output_img):
    import pandas as pd
    from scipy.stats import gaussian_kde

    df = pd.read_csv(csv_path)
    df = df[
        (df["error"] == "")
        & (df["operator_time"] > 0)
        & (df["n_verts"] >= MIN_VERTS_FOR_OPS)
    ]

    methods = df["method"].unique()
    cmap = plt.cm.get_cmap("tab10", len(methods))
    color_map = {m: cmap(i) for i, m in enumerate(sorted(methods))}

    fig, ax = plt.subplots(figsize=(14, 8))

    for method in sorted(methods):
        sub = df[df["method"] == method]
        if len(sub) < 2:
            continue
        c = color_map[method]
        ax.scatter(
            sub["n_verts"],
            sub["operator_time"],
            s=8,
            alpha=0.35,
            color=c,
            label=method,
            zorder=2,
        )

        if len(sub) > 10:
            xs = sub["n_verts"].values
            ys = sub["operator_time"].values

            xs_min, xs_max = xs.min(), xs.max()
            x_grid = np.linspace(xs_min, xs_max, 200)

            try:
                kde = gaussian_kde(np.vstack([xs, ys]))
                y_grid = np.linspace(ys.min(), ys.max(), 200)
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = np.reshape(kde(positions), X.shape)

                ax.contour(
                    X, Y, Z, levels=5, colors=[c], alpha=0.6, linewidths=1.2, zorder=3
                )
            except Exception:
                pass

            try:
                n_bins = min(30, max(5, len(sub) // 20))
                bins = pd.cut(xs, bins=n_bins)
                grouped = (
                    sub.groupby(bins, observed=True)
                    .agg(
                        n_v_c=("n_verts", "mean"),
                        t_mean=("operator_time", "mean"),
                        t_std=("operator_time", "std"),
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
                        alpha=0.12,
                        zorder=1,
                    )
            except Exception:
                pass

    ax.set_xlabel("Number of Vertices", fontsize=13)
    ax.set_ylabel("Spectral Operator Time (s)", fontsize=13)
    ax.set_title("Spectral Operator Time Distribution by Method", fontsize=14)
    ax.legend(fontsize=10, markerscale=3)
    ax.grid(True, alpha=0.25)
    ax.set_yscale("log")
    fig.tight_layout()

    out_path = FIG_DIR / output_img
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Plot saved to {out_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(
        description="Benchmark spectral operator time across methods and plot distribution."
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--methods",
        type=str,
        default="nanoshaper_gs0.3,nanoshaper_gs0.4,nanoshaper_gs0.6,alpha,edtsurf",
        help="Comma-separated method labels. NanoShaper: nanoshaper_gs<scale>, others: alpha, edtsurf",
    )
    parser.add_argument(
        "--output-csv", type=str, default="operator_time_distribution_raw.csv"
    )
    parser.add_argument(
        "--output-img", type=str, default="operator_time_distribution.png"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip benchmarking, just plot from existing CSV",
    )
    args = parser.parse_args()

    if args.plot_only:
        plot_distribution(FIG_DIR / args.output_csv, args.output_img)
        return

    methods = [m.strip() for m in args.methods.split(",")]

    holo_stems = {p.stem.replace("_holo", "") for p in Path(PDB_DIR).glob("*_holo.pdb")}
    pdb_files = sorted(
        str(p) for p in Path(PDB_DIR).glob("*.pdb") if p.stem not in holo_stems
    )

    print(f"PDBs: {len(pdb_files)}")
    print(f"Methods: {methods}")
    print(f"Workers: {args.workers}")

    csv_path = FIG_DIR / args.output_csv
    all_rows = []
    wall_clock = {}
    ctx = multiprocessing.get_context("spawn")

    with ctx.Pool(args.workers) as pool:
        for method in methods:
            task_args = [(p, method) for p in pdb_files]
            n = len(task_args)
            print(f"\n  {method} ({n} tasks, {args.workers} workers)", flush=True)

            method_rows = []
            t_wall_start = time.time()
            for i, row in enumerate(pool.imap_unordered(process_one, task_args), 1):
                method_rows.append(row)
                if i % 100 == 0:
                    print(
                        f"    {i}/{n} done ({time.time() - t_wall_start:.1f}s)",
                        flush=True,
                    )
            wc = time.time() - t_wall_start
            print(f"    {n}/{n} done -- wall clock: {wc:.1f}s", flush=True)

            wall_clock[method] = wc
            all_rows.extend(method_rows)
            save_csv(all_rows, csv_path)

    print(f"Saved {len(all_rows)} rows to {csv_path}")

    print("\n" + "=" * 110)
    print(
        f"{'Method':>20} | {'N':>5} | {'Avg Verts':>10} | {'Avg Op Time (s)':>16} | "
        f"{'Wall Clock':>11} | {'Time/Prot':>11}"
    )
    print("-" * 110)
    for method in methods:
        valid = [
            r
            for r in all_rows
            if r["method"] == method and r["operator_time"] > 0 and r["error"] == ""
        ]
        if not valid:
            wc = wall_clock.get(method, 0.0)
            print(
                f"{method:>20} | {0:>5} | {'--':>10} | {'--':>16} | {wc:9.1f}s | {'--':>11}"
            )
            continue
        avg_v = np.mean([r["n_verts"] for r in valid])
        avg_t = np.mean([r["operator_time"] for r in valid])
        wc = wall_clock.get(method, 0.0)
        t_per = wc / len(valid) if len(valid) > 0 else 0.0
        print(
            f"{method:>20} | {len(valid):>5} | {avg_v:10.1f} | {avg_t:16.4f} | "
            f"{wc:9.1f}s | {t_per:9.4f}s"
        )
    print("=" * 110)
    total_wall = sum(wall_clock.values())
    print(f"Total wall clock: {total_wall:.1f}s ({total_wall / 60:.1f}min)")
    print(f"Total proteins: {len(pdb_files)}")

    plot_distribution(csv_path, args.output_img)


if __name__ == "__main__":
    main()
