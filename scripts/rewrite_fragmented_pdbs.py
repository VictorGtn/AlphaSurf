"""Rewrite fragmented-side PDBs in place, keeping only the interface CC.

For every side listed in ``interface_cc.csv`` with ``n_components > 1``,
parse the original PDB, recompute all-atom CC labels (same 5.0 A radius graph
as ``extract_interface_cc.py``), and rewrite the file keeping only ATOM
records whose atom index falls in ``interface_cc_label``. Original file is
preserved as ``<stem>.orig.pdb`` (hardlinked before write).
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDB_DIR = REPO_ROOT / "data/pinder-pair/pdb"
DEFAULT_CSV = REPO_ROOT / "data/pinder-pair/interface_cc.csv"

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402
from scripts.extract_interface_cc import (  # noqa: E402
    SPLIT_TAGS,
    count_components_all_atom,
)

SPLIT_TO_TAG = {split: tag for _, (tag, split) in SPLIT_TAGS.items()}


def iter_fragmented_sides(csv_path):
    """Yield ``(system, tag, split, side, iface_cc_label)`` for each fragmented side."""
    grouped = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["side"] not in ("R", "L"):
                continue
            try:
                n_components = int(row["n_components"])
                iface_cc_label = int(row["interface_cc_label"])
            except ValueError:
                continue
            grouped[(row["system"], row["split"])][row["side"]] = (
                n_components,
                iface_cc_label,
            )
    for (system, split), sides in grouped.items():
        tag = SPLIT_TO_TAG.get(split, "")
        for side, (n_components, iface_cc_label) in sides.items():
            if n_components > 1:
                yield system, tag, split, side, iface_cc_label


def rewrite_side(pdb_path, iface_cc_label, graph_cutoff=5.0):
    """Hardlink ``<stem>.pdb`` to ``<stem>.orig.pdb`` then rewrite in place.

    If the backup already exists, parses the backup instead of the live file
    so the operation is idempotent across re-runs. Returns
    ``(n_atoms_in, n_atoms_kept)``. Raises on any mismatch between
    parser-atom count and filtered ATOM-line count.
    """
    backup = pdb_path.with_name(pdb_path.stem + ".orig.pdb")
    source = backup if backup.exists() else pdb_path

    parsed = parse_pdb_path(os.path.abspath(source), use_pqr=False)
    if parsed is None or parsed[5] is None:
        raise RuntimeError("parse_pdb_path returned None")
    atom_pos = np.asarray(parsed[5], dtype=np.float64)
    n_atoms = len(atom_pos)
    if n_atoms == 0:
        raise RuntimeError("no atoms parsed")

    _, labels = count_components_all_atom(atom_pos, cutoff=graph_cutoff)
    keep_mask = labels == iface_cc_label
    if not keep_mask.any():
        raise RuntimeError(
            f"interface_cc_label {iface_cc_label} not present "
            f"(labels in [0, {int(labels.max())}])"
        )

    kept_lines = []
    atom_idx = 0
    with open(source) as f:
        for line in f:
            if line[0:6].strip() != "ATOM":
                continue
            name = line[12:16].strip()
            if name.startswith("H"):
                continue
            if atom_idx >= n_atoms:
                raise RuntimeError(
                    f"file has more ATOM records than parser returned ({n_atoms})"
                )
            if keep_mask[atom_idx]:
                kept_lines.append(line)
            atom_idx += 1
    if atom_idx != n_atoms:
        raise RuntimeError(
            f"parser atom count ({n_atoms}) != filtered file ATOM count ({atom_idx})"
        )

    if not backup.exists():
        os.link(pdb_path, backup)

    tmp = pdb_path.with_name(pdb_path.name + ".tmp")
    with open(tmp, "w") as f:
        f.writelines(kept_lines)
    tmp.replace(pdb_path)
    return n_atoms, int(keep_mask.sum())


def process_one(args_tuple):
    system, tag, split, side, iface_cc_label, pdb_dir, dry_run = args_tuple
    stem = f"{system}_{side}{tag}"
    pdb_path = pdb_dir / f"{stem}.pdb"
    if not pdb_path.exists():
        return (system, tag, split, side), None, f"missing: {pdb_path.name}"
    if dry_run:
        backup = pdb_path.with_name(pdb_path.stem + ".orig.pdb")
        source = backup if backup.exists() else pdb_path
        parsed = parse_pdb_path(os.path.abspath(source), use_pqr=False)
        if parsed is None or parsed[5] is None:
            return (system, tag, split, side), None, "parse_pdb_path returned None"
        atom_pos = np.asarray(parsed[5], dtype=np.float64)
        _, labels = count_components_all_atom(atom_pos)
        n_keep = int((labels == iface_cc_label).sum())
        return (
            (system, tag, split, side),
            {
                "n_atoms_in": len(atom_pos),
                "n_atoms_kept": n_keep,
                "wrote": False,
            },
            None,
        )
    try:
        n_in, n_keep = rewrite_side(pdb_path, iface_cc_label)
    except Exception as e:
        return (system, tag, split, side), None, f"{type(e).__name__}: {e}"
    return (
        (system, tag, split, side),
        {
            "n_atoms_in": n_in,
            "n_atoms_kept": n_keep,
            "wrote": True,
        },
        None,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb-dir", default=str(DEFAULT_PDB_DIR))
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Recompute counts only; do not write or backup.",
    )
    args = ap.parse_args()

    pdb_dir = Path(args.pdb_dir)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"missing CSV: {csv_path}")

    tasks = [
        (s, t, sp, sd, icc, pdb_dir, args.dry_run)
        for (s, t, sp, sd, icc) in iter_fragmented_sides(csv_path)
    ]
    print(
        f"Fragmented sides: {len(tasks)}  Workers: {args.workers}"
        f"{'  (dry-run)' if args.dry_run else ''}",
        file=sys.stderr,
    )

    n_ok = n_err = n_dropped = 0
    atoms_in_total = atoms_keep_total = 0
    with mp.Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(process_one, tasks, chunksize=8)
        for i, (key, result, error) in enumerate(results, 1):
            if error:
                n_err += 1
                print(f"  ERR {key}: {error}", file=sys.stderr)
                continue
            n_ok += 1
            atoms_in_total += result["n_atoms_in"]
            atoms_keep_total += result["n_atoms_kept"]
            n_dropped += result["n_atoms_in"] - result["n_atoms_kept"]
            if i % 500 == 0 or i == len(tasks):
                print(
                    f"  [{i}/{len(tasks)}] ok={n_ok} err={n_err} "
                    f"dropped={n_dropped} atoms",
                    file=sys.stderr,
                )

    print(
        f"Done. ok={n_ok} err={n_err} "
        f"atoms_kept={atoms_keep_total}/{atoms_in_total} "
        f"dropped={n_dropped}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
