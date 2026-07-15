"""Preprocess MISATO for residue-level ligand binding-site prediction.

The ICML 2025 static-dynamic-fusion implementation labels a residue positive
when its C-alpha is at most 10 Angstrom from any ligand heavy atom.  The raw
MISATO MD file contains hydrogens and ``trajectory_coordinates``; this script
uses frame 0 as the reference conformation and strips hydrogens, matching the
processed ``adaptability_MD.hdf5`` consumed by the reference implementation.

Inputs::

    data_dir/MD.hdf5
    data_dir/splits/{train,val,test}.txt

Outputs::

    data_dir/binding_site/<pdb_id>.pt

Each output contains the reference protein atom cloud, atom metadata, C-alpha
indices and residue types, and fixed binary labels. Trajectory coordinates are
read lazily from MD.hdf5 during training rather than duplicated on disk.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Codes from MISATO's published atoms_type_map / atoms_residue_map.
CA_ATOM_TYPE = 14  # AMBER CX, used for protein C-alpha atoms
MISATO_RESIDUES = {
    2: "ALA",
    3: "ARG",
    4: "ASN",
    5: "ASP",
    6: "CYS",
    7: "CYS",
    8: "GLN",
    9: "GLU",
    10: "GLY",
    11: "HIS",
    12: "ILE",
    13: "LEU",
    14: "LYS",
    15: "MET",
    16: "PHE",
    17: "PRO",
    18: "SER",
    19: "THR",
    20: "TRP",
    21: "TYR",
    22: "VAL",
}

# AlphaSurf residue ordering (alphasurf.protein.graphs.res_type_dict).
ALPHASURF_RESIDUES = {
    "ALA": 0,
    "GLY": 1,
    "SER": 2,
    "THR": 3,
    "LEU": 4,
    "ILE": 5,
    "VAL": 6,
    "ASN": 7,
    "GLN": 8,
    "ARG": 9,
    "HIS": 10,
    "TRP": 11,
    "PHE": 12,
    "TYR": 13,
    "GLU": 14,
    "ASP": 15,
    "LYS": 16,
    "PRO": 17,
    "CYS": 18,
    "MET": 19,
    "UNK": 20,
}

# AMBER atom-type code -> element. Surface generation only needs a VdW radius;
# atom names are not required. Codes 1..33 are protein types, 34..110 ligand.
ATOM_TYPE_TO_ELEMENT = {
    1: "C",
    2: "C",
    3: "C",
    4: "C",
    5: "C",
    6: "C",
    7: "C",
    8: "C",
    9: "C",
    10: "C",
    11: "C",
    12: "C",
    13: "C",
    14: "C",
    15: "H",
    16: "H",
    17: "H",
    18: "H",
    19: "H",
    20: "H",
    21: "H",
    22: "H",
    23: "H",
    24: "N",
    25: "N",
    26: "N",
    27: "N",
    28: "N",
    29: "O",
    30: "O",
    31: "O",
    32: "S",
    33: "S",
}
VDW_RADIUS = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--md-path", default=None)
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--label-cutoff", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Debug option: process at most this many structures",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Number of independent HDF5 reader processes",
    )
    return parser.parse_args()


def load_split_ids(splits_dir: str) -> dict[str, list[str]]:
    splits = {}
    for split in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{split}.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing official MISATO split: {path}. Run download_misato.sh."
            )
        with open(path) as handle:
            splits[split] = [line.strip() for line in handle if line.strip()]
    return splits


def _lookup_h5_key(md: h5py.File, pdb_id: str) -> str | None:
    for candidate in (pdb_id, pdb_id.lower(), pdb_id.upper()):
        if candidate in md:
            return candidate
    return None


def preprocess_entry(group: h5py.Group, cutoff: float) -> dict:
    if "atoms_coordinates_ref" in group:
        coords = np.asarray(group["atoms_coordinates_ref"], dtype=np.float32)
    elif "trajectory_coordinates" in group:
        coords = np.asarray(group["trajectory_coordinates"][0], dtype=np.float32)
    else:
        raise KeyError("missing atoms_coordinates_ref and trajectory_coordinates")
    atom_types = np.asarray(group["atoms_type"], dtype=np.int64)
    atom_elements = np.asarray(group["atoms_element"], dtype=np.int64)
    residue_codes = np.asarray(group["atoms_residue"], dtype=np.int64)
    molecule_starts = np.asarray(group["molecules_begin_atom_index"], dtype=np.int64)
    if molecule_starts.size == 0:
        raise ValueError("molecules_begin_atom_index is empty")

    ligand_start = int(molecule_starts[-1])
    protein_source_index = np.flatnonzero(atom_elements[:ligand_start] != 1)
    ligand_source_index = ligand_start + np.flatnonzero(
        atom_elements[ligand_start:] != 1
    )
    protein_coords = coords[protein_source_index]
    protein_atom_types = atom_types[protein_source_index]
    protein_residue_codes = residue_codes[protein_source_index]
    ligand_coords = coords[ligand_source_index]
    if len(protein_coords) == 0 or len(ligand_coords) == 0:
        raise ValueError("empty protein or ligand")

    ca_local = np.flatnonzero(protein_atom_types == CA_ATOM_TYPE)
    if len(ca_local) == 0:
        raise ValueError("no C-alpha atoms")
    ca_coords = protein_coords[ca_local]
    ca_residue_codes = protein_residue_codes[ca_local]
    residue_types = np.asarray(
        [
            ALPHASURF_RESIDUES.get(MISATO_RESIDUES.get(int(code), "UNK"), 20)
            for code in ca_residue_codes
        ],
        dtype=np.int64,
    )

    # Chunking avoids materializing a potentially large all-pairs matrix.
    min_dist2 = np.full(len(ca_coords), np.inf, dtype=np.float32)
    for start in range(0, len(ligand_coords), 512):
        ligand_chunk = ligand_coords[start : start + 512]
        dist2 = ((ca_coords[:, None, :] - ligand_chunk[None, :, :]) ** 2).sum(-1)
        min_dist2 = np.minimum(min_dist2, dist2.min(axis=1))
    labels = (min_dist2 <= cutoff * cutoff).astype(np.int64)

    radii = np.asarray(
        [
            VDW_RADIUS.get(ATOM_TYPE_TO_ELEMENT.get(int(code), "C"), 1.70)
            for code in protein_atom_types
        ],
        dtype=np.float32,
    )

    return {
        "atom_pos": torch.from_numpy(protein_coords),
        "atom_radius": torch.from_numpy(radii),
        "ca_pos": torch.from_numpy(ca_coords),
        "ca_atom_index": torch.from_numpy(ca_local.astype(np.int64)),
        # Indices into the raw HDF5 trajectory. Dataset workers use these to
        # load the same heavy-atom subset for any sampled MD frame.
        "protein_source_index": torch.from_numpy(protein_source_index.astype(np.int64)),
        "ligand_start": ligand_start,
        "residue_type": torch.from_numpy(residue_types),
        "y": torch.from_numpy(labels),
        "label_cutoff": float(cutoff),
    }


_WORKER_MD = None


def _init_worker(md_path: str):
    """Open one read-only HDF5 handle per process (h5py handles are not shared)."""
    global _WORKER_MD
    _WORKER_MD = h5py.File(md_path, "r")


def _process_one(job):
    """Process and atomically save one complex."""
    pdb_id, out_path, cutoff = job
    key = _lookup_h5_key(_WORKER_MD, pdb_id)
    if key is None:
        return "failed", pdb_id, "absent from MD.hdf5"
    tmp_path = f"{out_path}.tmp.{os.getpid()}"
    try:
        item = preprocess_entry(_WORKER_MD[key], cutoff)
        item["pdb_id"] = pdb_id.lower()
        torch.save(item, tmp_path)
        os.replace(tmp_path, out_path)
        return "processed", pdb_id, None
    except Exception as exc:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        return "failed", pdb_id, str(exc)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    data_dir = Path(args.data_dir)
    md_path = Path(args.md_path or data_dir / "MD.hdf5")
    splits_dir = Path(args.splits_dir or data_dir / "splits")
    output_dir = Path(args.output_dir or data_dir / "binding_site")
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_split_ids(str(splits_dir))
    all_ids = list(dict.fromkeys(pid for split in splits.values() for pid in split))
    if args.max_samples is not None:
        all_ids = all_ids[: args.max_samples]

    jobs = []
    skipped = 0
    for pdb_id in all_ids:
        out_path = output_dir / f"{pdb_id.lower()}.pt"
        if out_path.exists() and not args.overwrite:
            skipped += 1
        else:
            jobs.append((pdb_id, str(out_path), args.label_cutoff))

    num_workers = max(1, min(args.num_workers, len(jobs) or 1))
    logger.info(
        "Starting %d worker(s): pending=%d existing=%d total=%d",
        num_workers,
        len(jobs),
        skipped,
        len(all_ids),
    )
    processed = failed = 0
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_worker,
        initargs=(str(md_path),),
    ) as executor:
        for completed, result in enumerate(
            executor.map(_process_one, jobs, chunksize=1), 1
        ):
            status, pdb_id, error = result
            if status == "processed":
                processed += 1
            else:
                failed += 1
                logger.warning("Failed %s: %s", pdb_id, error)
            if completed % 250 == 0 or completed == len(jobs):
                logger.info(
                    "Completed %d/%d pending (new=%d failed=%d existing=%d)",
                    completed,
                    len(jobs),
                    processed,
                    failed,
                    skipped,
                )

    logger.info("Done: new=%d existing=%d failed=%d", processed, skipped, failed)


if __name__ == "__main__":
    main()
