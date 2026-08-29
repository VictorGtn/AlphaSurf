#!/usr/bin/env python3
"""
Per-PDB all-atom connected-component analysis.

For each .pdb in --pdb-dir, builds a radius-neighbor graph on atomic
positions (cutoff A) and runs scipy.sparse.csgraph.connected_components.

Output columns:
  pdb_name, n_atoms, n_components, component_sizes
"""

import argparse
import csv
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402


def count_components_all_atom(atom_pos, cutoff=5.0):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(atom_pos)
    if n == 0:
        return 0, np.array([], dtype=np.int64)
    tree = cKDTree(atom_pos)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n)
        )
    else:
        graph = coo_matrix((n, n), dtype=np.int8)
    n_comp, labels = connected_components(csgraph=graph, directed=False)
    sizes = np.bincount(labels)
    return int(n_comp), np.sort(sizes)[::-1].astype(np.int64)


def _sizes_str(sizes):
    return ";".join(str(int(s)) for s in sizes)


def analyze_one(args):
    pdb_path, cutoff = args
    pdb_name = os.path.basename(pdb_path)
    row = {
        "pdb_name": pdb_name,
        "n_atoms": 0,
        "n_components": 0,
        "component_sizes": "",
        "error": "",
    }
    try:
        parsed = parse_pdb_path(os.path.abspath(pdb_path), use_pqr=False)
        if parsed is None or parsed[5] is None:
            row["error"] = "parse_pdb_path returned None"
            return row
        atom_pos = np.asarray(parsed[5], dtype=np.float64)
        row["n_atoms"] = len(atom_pos)
        if len(atom_pos) == 0:
            row["error"] = "empty atom array"
            return row
        n_comp, sizes = count_components_all_atom(atom_pos, cutoff=cutoff)
        row["n_components"] = n_comp
        row["component_sizes"] = _sizes_str(sizes)
    except Exception as e:
        row["error"] = str(e)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="All-atom connected-component analysis on PDB files"
    )
    parser.add_argument(
        "--pdb-dir", required=True, help="Directory containing .pdb files"
    )
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Angstrom cutoff for atom-neighbor graph (default 5.0)",
    )
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "-w", "--workers", type=int, default=1, help="Parallel workers (0 = all CPUs)"
    )
    args = parser.parse_args()

    pdb_files = sorted(str(x) for x in Path(args.pdb_dir).glob("*.pdb"))
    print(f"PDB files found: {len(pdb_files)}")
    if args.max_files:
        pdb_files = pdb_files[: args.max_files]
        print(f"Limited to {args.max_files} files")
    if not pdb_files:
        return

    task_args = [(p, args.cutoff) for p in pdb_files]
    n_workers = args.workers if args.workers > 0 else multiprocessing.cpu_count()

    rows = []
    t0 = time.time()
    # Log-friendly tqdm: mininterval=2s so a SLURM .log gets ~1 line per 2s
    # instead of one per \r refresh. Plain bar_format avoids ASCII bar spam.
    bar_fmt = "{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    if n_workers == 1:
        with tqdm(
            total=len(task_args),
            desc="analyze",
            unit="file",
            mininterval=2.0,
            bar_format=bar_fmt,
        ) as pbar:
            for a in task_args:
                rows.append(analyze_one(a))
                pbar.update(1)
    else:
        with (
            multiprocessing.Pool(n_workers) as pool,
            tqdm(
                total=len(task_args),
                desc="analyze",
                unit="file",
                mininterval=2.0,
                bar_format=bar_fmt,
            ) as pbar,
        ):
            for row in pool.imap_unordered(analyze_one, task_args, chunksize=8):
                rows.append(row)
                pbar.update(1)

    rows.sort(key=lambda r: r["pdb_name"])
    fields = ["pdb_name", "n_atoms", "n_components", "component_sizes", "error"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n_err = sum(1 for r in rows if r["error"])
    multi = sum(1 for r in rows if r["n_components"] >= 2)
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Total: {len(rows)}  with >=2 components: {multi}  errors: {n_err}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
