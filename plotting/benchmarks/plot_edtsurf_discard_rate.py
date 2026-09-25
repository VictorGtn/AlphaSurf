#!/usr/bin/env python3
"""
Plot EDTSurf Discard Rate vs. Average Vertices across Grid Scales.

Computes the number of surfaces discarded due to multiple connected components
(>1 component larger than 1% of the maximum component) and the average number
of vertices, plotted against the grid scale.

Uses multiprocessing to evaluate EDTSurf across grid scales 0.1-0.5.
"""

import argparse
import csv
import multiprocessing
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh

project_root = str(Path(__file__).resolve().parents[2])
FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "benchmarks"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

EDTSURF_BIN = str(Path(project_root).parent / "EDTSurf" / "EDTSurf")
PDB_DIR = os.path.join(project_root, "data", "pinder-pair", "pdb")


def _extract_parse_time(stdout):
    for line in stdout.splitlines():
        if line.startswith("Parse time"):
            try:
                return float(line.split()[2])
            except (IndexError, ValueError):
                pass
    return None


def try_edtsurf(pdb_path, grid_scale=0.5, probe_radius=1.4):
    out_base = os.path.join(tempfile.gettempdir(), f"edtsurf_disc_{os.getpid()}")
    ply_file = out_base + ".ply"

    try:
        result = subprocess.run(
            [
                EDTSURF_BIN,
                "-i",
                pdb_path,
                "-o",
                out_base,
                "-s",
                "3",
                "-p",
                str(probe_radius),
                "-f",
                str(grid_scale),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        parse_time = _extract_parse_time(result.stdout or "")

        if not os.path.exists(ply_file):
            return None, f"no_output(exit={result.returncode})", parse_time

        mesh = trimesh.load(ply_file, process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        return (verts, faces), None, parse_time
    except subprocess.TimeoutExpired:
        return None, "timeout", None
    except Exception as e:
        return None, str(e)[:80], None
    finally:
        for ext in [".ply", ".asa", "-cav.pdb"]:
            p = out_base + ext
            if os.path.exists(p):
                os.remove(p)


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


def _timeout_handler(signum, frame):
    raise TimeoutError("process_one timed out")


def process_one(args):
    import signal

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(360)

    pdb_path, grid_scale = args
    name = Path(pdb_path).stem

    try:
        crash = False
        multi_comp = False
        n_verts = 0
        parse_time = None

        res, err, parse_time = try_edtsurf(pdb_path, grid_scale=grid_scale)
        if err:
            crash = True
        else:
            _, n_verts, _, multi_comp = count_components(*res)
    except TimeoutError:
        crash = True
        multi_comp = False
        n_verts = 0
    finally:
        signal.alarm(0)

    return {
        "pdb": name,
        "grid_scale": grid_scale,
        "crash": crash,
        "multi_comp": multi_comp,
        "verts": n_verts,
        "parse_time": parse_time,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--grid-scales",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated grid scales",
    )
    parser.add_argument(
        "--output-img",
        type=str,
        default="edtsurf_discard_rate_vs_vertices.png",
    )
    args = parser.parse_args()

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
    y_parse_time = []

    for gs in sorted(results_by_scale.keys()):
        rs = results_by_scale[gs]
        discarded_count = sum(1 for r in rs if r["multi_comp"] and not r["crash"])
        discarded_pct = 100.0 * discarded_count / len(rs) if rs else 0
        valid_verts = [r["verts"] for r in rs if not r["crash"]]
        avg_verts = np.mean(valid_verts) if valid_verts else 0
        parse_times = [r["parse_time"] for r in rs if r["parse_time"] is not None]
        avg_parse_ms = np.mean(parse_times) * 1000 if parse_times else 0

        x_scales.append(gs)
        y_discarded.append(discarded_pct)
        y_vertices.append(avg_verts)
        y_parse_time.append(avg_parse_ms)

    print("\n--- Summary ---")
    print(
        f"{'Grid Scale':>10} | {'Discarded %':>11} | {'Avg Vertices':>12} | {'Parse (ms)':>10}"
    )
    for gs, disc, verts, pt in zip(x_scales, y_discarded, y_vertices, y_parse_time):
        print(f"{gs:10.2f} | {disc:10.2f}% | {verts:12.1f} | {pt:10.1f}")

    summary_csv = FIG_DIR / "edtsurf_discard_rate_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Grid Scale", "Discarded %", "Avg Vertices", "Parse (ms)"])
        for gs, disc, verts, pt in zip(x_scales, y_discarded, y_vertices, y_parse_time):
            writer.writerow([gs, disc, verts, pt])
    print(f"\nSummary data saved to {summary_csv}")

    raw_csv = FIG_DIR / "edtsurf_discard_rate_raw.csv"
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

    plt.title("EDTSurf: Discarded Surfaces and Avg Vertices vs Grid Scale", fontsize=14)
    plt.tight_layout()

    out_path = FIG_DIR / args.output_img
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
