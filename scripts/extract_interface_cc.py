"""Find the all-atom connected component containing the dist-mode binding interface.

For each pinder system in ``data/pinder-pair/pdb/``, parse ``<system>_R.pdb`` and
``<system>_L.pdb`` with ``parse_pdb_path`` (same loader as
``pdb_atom_components.py``), build a radius-neighbor graph on atom positions,
and compute the dist-mode interface
(``cdist(atom_pos_R, atom_pos_L) < threshold``). For any fragmented graph,
record which component holds the interface atoms.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDB_DIR = REPO_ROOT / "data/pinder-pair/pdb"

SPLIT_TAGS = {
    "systems_train.csv": ("", "train"),
    "systems_val.csv": ("", "val"),
    "systems_test_holo.csv": ("_holo", "test_holo"),
    "systems_test_apo.csv": ("_apo", "test_apo"),
    "systems_test_af2.csv": ("_af2", "test_af2"),
}

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alphasurf.protein.graphs import parse_pdb_path  # noqa: E402


def pinder_opened_systems(csv_dir, pdb_dir):
    """Yield ``(system, tag, split)`` for every system in any ``systems_*.csv``
    whose tagged PDB pair exists. Train/val use bare ``<system>_R.pdb`` /
    ``<system>_L.pdb``; test splits use ``_holo``/``_apo``/``_af2`` suffixes.
    Systems appearing in multiple test splits are emitted once per split."""
    opened, skipped = [], []
    seen_per_split = {split: set() for _, split in SPLIT_TAGS.values()}
    for csv_name, (tag, split) in SPLIT_TAGS.items():
        csv_path = csv_dir / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                system = row.get("id")
                if not system or system in seen_per_split[split]:
                    continue
                seen_per_split[split].add(system)
                receptor_path = pdb_dir / f"{system}_R{tag}.pdb"
                ligand_path = pdb_dir / f"{system}_L{tag}.pdb"
                if receptor_path.exists() and ligand_path.exists():
                    opened.append((system, tag, split))
                else:
                    skipped.append((system, tag, split))
    return opened, skipped


def count_components_all_atom(atom_pos, cutoff=5.0):
    """Same function as ``pdb_atom_components.count_components_all_atom``."""
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
    return int(n_comp), labels


def compute_interface(pos_1, pos_2, threshold):
    """Dist-mode interface, mirroring PinderPairDataset._compute_interface."""
    from scipy.spatial.distance import cdist

    dists = cdist(pos_1, pos_2)
    mask = dists < threshold
    i_idx, j_idx = np.where(mask)
    return np.stack([i_idx, j_idx])


def process_system(args_tuple):
    system, tag, split, pdb_dir, graph_cutoff, iface_threshold = args_tuple
    r_path = pdb_dir / f"{system}_R{tag}.pdb"
    l_path = pdb_dir / f"{system}_L{tag}.pdb"
    if not r_path.exists() or not l_path.exists():
        return (
            (system, tag, split),
            None,
            f"missing files: {r_path.name}, {l_path.name}",
        )

    try:
        parsed_r = parse_pdb_path(os.path.abspath(r_path), use_pqr=False)
        parsed_l = parse_pdb_path(os.path.abspath(l_path), use_pqr=False)
    except Exception as e:
        return (system, tag, split), None, f"parse_pdb_path error: {e}"
    if (
        parsed_r is None
        or parsed_r[5] is None
        or parsed_l is None
        or parsed_l[5] is None
    ):
        return (system, tag, split), None, "parse_pdb_path returned None"

    atom_pos_r = np.asarray(parsed_r[5], dtype=np.float64)
    atom_pos_l = np.asarray(parsed_l[5], dtype=np.float64)
    if len(atom_pos_r) == 0 or len(atom_pos_l) == 0:
        return (
            (system, tag, split),
            None,
            f"empty atom array: R={len(atom_pos_r)} L={len(atom_pos_l)}",
        )

    pairs = compute_interface(atom_pos_r, atom_pos_l, iface_threshold)
    interface_r = set(int(i) for i in pairs[0])
    interface_l = set(int(j) for j in pairs[1])

    labels_r = labels_l = None
    out_base = {}
    comp_sizes_per_side = {}
    comp_iface_counts_per_side = {}
    for side, atom_pos, iface in (
        ("R", atom_pos_r, interface_r),
        ("L", atom_pos_l, interface_l),
    ):
        n_atoms = int(len(atom_pos))
        n_comp, labels = count_components_all_atom(atom_pos, cutoff=graph_cutoff)
        comp_sizes = Counter(labels.tolist())
        largest_label, largest_size = max(
            comp_sizes.items(), key=lambda kv: (kv[1], -kv[0])
        )
        comp_iface_counts = Counter()
        for node in iface:
            if 0 <= node < n_atoms:
                comp_iface_counts[labels[node]] += 1

        out_base[side] = {
            "n_atoms": n_atoms,
            "n_components": n_comp,
            "largest_cc_label": int(largest_label),
            "largest_cc_size": int(largest_size),
            "n_interface_atoms": len(iface),
        }
        comp_sizes_per_side[side] = comp_sizes
        comp_iface_counts_per_side[side] = comp_iface_counts
        if side == "R":
            labels_r = labels
        else:
            labels_l = labels

    # Pick the (R-CC, L-CC) pair jointly: maximize interface atom pairs kept.
    # With no interface, fall back to each side's largest CC.
    if len(pairs[0]) > 0:
        K_r = int(labels_r.max()) + 1
        K_l = int(labels_l.max()) + 1
        pair_counts = np.zeros((K_r, K_l), dtype=np.int64)
        np.add.at(pair_counts, (labels_r[pairs[0]], labels_l[pairs[1]]), 1)
        r_cc, l_cc = (
            int(x) for x in np.unravel_index(pair_counts.argmax(), pair_counts.shape)
        )
    else:
        r_cc = out_base["R"]["largest_cc_label"]
        l_cc = out_base["L"]["largest_cc_label"]

    out_r = dict(out_base["R"])
    out_l = dict(out_base["L"])
    out_r["interface_cc_label"] = r_cc
    out_l["interface_cc_label"] = l_cc
    out_r["interface_cc_size"] = int(comp_sizes_per_side["R"][r_cc])
    out_l["interface_cc_size"] = int(comp_sizes_per_side["L"][l_cc])
    out_r["n_interface_in_cc"] = int(comp_iface_counts_per_side["R"].get(r_cc, 0))
    out_l["n_interface_in_cc"] = int(comp_iface_counts_per_side["L"].get(l_cc, 0))

    r_cc_pos = atom_pos_r[labels_r == r_cc]
    l_cc_pos = atom_pos_l[labels_l == l_cc]
    n_total = int(len(r_cc_pos) * len(l_cc_pos))
    if n_total > 0:
        from scipy.spatial.distance import cdist

        n_pos = int((cdist(r_cc_pos, l_cc_pos) < iface_threshold).sum())
    else:
        n_pos = 0
    for d in (out_r, out_l):
        d["n_iface_cc_pairs_total"] = n_total
        d["n_iface_cc_pairs_positive"] = n_pos
        d["n_iface_cc_pairs_negative"] = n_total - n_pos

    return (system, tag, split), {"R": out_r, "L": out_l}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", default=str(DEFAULT_PDB_DIR))
    parser.add_argument(
        "--systems", nargs="+", default=None, help="Optional explicit system stems."
    )
    parser.add_argument(
        "--from-pinder-csvs",
        default=None,
        help="If set, only process systems the pinder pipeline actually opens. "
        "Value is the directory containing systems_*.csv. Overrides glob.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=5.0,
        help="Angstrom cutoff for the atom-neighbor graph (default 5.0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Dist-mode interface threshold in Angstroms.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--fragmented-only",
        action="store_true",
        help="Only emit rows for graphs with >1 connected component.",
    )
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)

    if args.systems:
        tasks = [
            (s, "", "explicit", pdb_dir, args.cutoff, args.threshold)
            for s in args.systems
        ]
        skipped = []
    elif args.from_pinder_csvs:
        opened, skipped = pinder_opened_systems(Path(args.from_pinder_csvs), pdb_dir)
        print(f"Pinder CSVs: {args.from_pinder_csvs}", file=sys.stderr)
        print(f"Systems opened by pipeline: {len(opened)}", file=sys.stderr)
        if skipped:
            print(
                f"WARNING: {len(skipped)} CSV-referenced systems missing PDBs, e.g.:",
                file=sys.stderr,
            )
            for s, tag, split in skipped[:5]:
                print(f"  {s}{tag} ({split})", file=sys.stderr)
        tasks = [
            (s, tag, split, pdb_dir, args.cutoff, args.threshold)
            for (s, tag, split) in opened
        ]
    else:
        tasks = [
            (p.stem[:-2], "", "glob", pdb_dir, args.cutoff, args.threshold)
            for p in sorted(pdb_dir.glob("*_R.pdb"))
        ]
    print(f"Tasks: {len(tasks)}  Workers: {args.workers}", file=sys.stderr)
    print(
        f"Graph radius cutoff: {args.cutoff} A  "
        f"interface threshold: {args.threshold} A",
        file=sys.stderr,
    )

    fieldnames = [
        "system",
        "split",
        "side",
        "n_atoms",
        "n_components",
        "largest_cc_label",
        "largest_cc_size",
        "n_interface_atoms",
        "interface_cc_label",
        "interface_cc_size",
        "n_interface_in_cc",
        "n_iface_cc_pairs_total",
        "n_iface_cc_pairs_positive",
        "n_iface_cc_pairs_negative",
        "error",
    ]
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = n_err = n_rows = 0
    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        with mp.Pool(processes=args.workers) as pool:
            results = pool.imap_unordered(process_system, tasks, chunksize=8)
            for i, ((system, tag, split), result, error) in enumerate(results, 1):
                if error:
                    n_err += 1
                    writer.writerow(
                        {"system": system, "split": split, "side": "", "error": error}
                    )
                    continue
                n_ok += 1
                for side, stats in result.items():
                    if args.fragmented_only and stats["n_components"] <= 1:
                        continue
                    row = {"system": system, "split": split, "side": side, "error": ""}
                    row.update(stats)
                    writer.writerow(row)
                    n_rows += 1
                if i % 500 == 0:
                    f_out.flush()
                    print(
                        f"  [{i}/{len(tasks)}] ok={n_ok} err={n_err}", file=sys.stderr
                    )

    print(f"Done. ok={n_ok} err={n_err} rows={n_rows}", file=sys.stderr)
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
