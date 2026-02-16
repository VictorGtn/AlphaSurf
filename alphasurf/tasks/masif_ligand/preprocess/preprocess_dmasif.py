"""
Preprocess PDB files to create dMaSIF point cloud surfaces.

Uses DataLoader with multiple workers for parallel PDB parsing (CPU),
then processes on GPU in the main process.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Allow torch.load to resolve SurfaceObject
script_dir = os.path.dirname(os.path.realpath(__file__))
alphasurf_parent_dir = os.path.join(script_dir, "..", "..", "..")
sys.path.insert(0, alphasurf_parent_dir)

# Suppress warnings
warnings.filterwarnings("ignore")

from alphasurf.network_utils.misc_arch.dmasif_utils.geometry_processing import (
    atoms_to_points_normals,
)
from alphasurf.network_utils.misc_arch.timing import (
    enable_timing,
    print_timing_stats,
    time_operation,
)
from alphasurf.protein.graphs import parse_pdb_path
from alphasurf.protein.surfaces import SurfaceObject


def atom_types_to_onehot(
    atom_types: np.ndarray, num_atom_types: int = 22
) -> torch.Tensor:
    """Convert atom type indices to one-hot encoding."""
    N = len(atom_types)
    onehot = torch.zeros(N, num_atom_types, dtype=torch.float32)
    for i, atom_type_idx in enumerate(atom_types):
        if atom_type_idx < num_atom_types:
            onehot[i, atom_type_idx] = 1.0
        else:
            onehot[i, num_atom_types - 1] = 1.0
    return onehot


class PDBParsingDataset(Dataset):
    """
    Dataset that parses PDBs on CPU (in DataLoader workers).
    Returns parsed data that will be processed on GPU in main process.
    """

    def __init__(self, pdb_files, out_surf_dir, recompute=False):
        self.pdb_files = pdb_files
        self.out_surf_dir = out_surf_dir
        self.recompute = recompute

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        """Parse PDB file on CPU worker, return data for GPU processing."""
        pdb_path = self.pdb_files[idx]
        pdb_name = Path(pdb_path).stem
        out_surf_path = os.path.join(self.out_surf_dir, f"{pdb_name}.pt")

        # Check if already exists
        if os.path.exists(out_surf_path) and not self.recompute:
            return {
                "pdb_name": pdb_name,
                "skip": True,
                "out_path": out_surf_path,
            }

        try:
            # Parse PDB (CPU-bound, runs in worker process)
            arrays = parse_pdb_path(pdb_path, use_pqr=False)
            (
                amino_types,
                atom_chain_id,
                atom_amino_id,
                atom_names,
                atom_types,
                atom_pos,
                atom_charge,
                atom_radius,
                res_sse,
                amino_ids,
                atom_ids,
            ) = arrays

            # Convert to tensors (still on CPU)
            atom_coords = torch.from_numpy(atom_pos).float()
            atomtypes_onehot = atom_types_to_onehot(atom_types, num_atom_types=22)

            return {
                "pdb_name": pdb_name,
                "skip": False,
                "out_path": out_surf_path,
                "atom_coords": atom_coords,
                "atomtypes": atomtypes_onehot,
                "error": None,
            }
        except Exception as e:
            import traceback

            return {
                "pdb_name": pdb_name,
                "skip": False,
                "out_path": out_surf_path,
                "error": f"{str(e)}\n{traceback.format_exc()}",
            }


def collate_fn(batch):
    """Custom collate - just return the list of dicts."""
    return batch


def process_batch_on_gpu(
    batch,
    device,
    distance,
    smoothness,
    resolution,
    nits,
    sup_sampling,
    variance,
    log_file=None,
):
    """
    Process a batch of parsed PDBs on GPU using TRUE batching.
    Concatenates all atoms and uses batch vector for parallel processing.
    Returns list of (success, num_points) tuples.
    """
    results = [None] * len(batch)

    # Separate items: skip, error, or process
    to_process = []
    for i, item in enumerate(batch):
        pdb_name = item["pdb_name"]
        out_path = item["out_path"]

        if item["skip"]:
            try:
                existing = torch.load(out_path, weights_only=False, map_location="cpu")
                n_pts = existing.verts.shape[0] if existing.verts is not None else 0
                results[i] = (True, n_pts)
            except:
                results[i] = (True, None)
        elif item.get("error"):
            if log_file:
                with open(log_file, "a") as f:
                    f.write(f"{pdb_name}: {item['error']}\n")
            results[i] = (False, None)
        else:
            to_process.append((i, item))

    if not to_process:
        return results

    try:
        # Concatenate all atoms from batch (PyG-style batching)
        all_coords = []
        all_atomtypes = []
        batch_vector = []

        for batch_idx, (orig_idx, item) in enumerate(to_process):
            coords = item["atom_coords"]
            atomtypes = item["atomtypes"]
            all_coords.append(coords)
            all_atomtypes.append(atomtypes)
            batch_vector.append(torch.full((len(coords),), batch_idx, dtype=torch.long))

        # Stack and move to GPU
        all_coords = torch.cat(all_coords, dim=0).to(device)
        all_atomtypes = torch.cat(all_atomtypes, dim=0).to(device)
        batch_vector = torch.cat(batch_vector, dim=0).to(device)

        # Single batched GPU call!
        points, normals, batch_points = atoms_to_points_normals(
            all_coords,
            batch_vector,
            num_atoms=22,
            distance=distance,
            smoothness=smoothness,
            resolution=resolution,
            nits=nits,
            atomtypes=all_atomtypes,
            sup_sampling=sup_sampling,
            variance=variance,
        )

        # Move results to CPU
        points = points.cpu()
        normals = normals.cpu()
        batch_points = batch_points.cpu()

        # Split results by protein and save
        for batch_idx, (orig_idx, item) in enumerate(to_process):
            mask = batch_points == batch_idx
            protein_points = points[mask]
            protein_normals = normals[mask]
            protein_batch = torch.zeros(protein_points.shape[0], dtype=torch.long)

            surface = SurfaceObject(
                verts=protein_points,
                faces=None,
                vnormals=protein_normals,
                batch=protein_batch,
            )

            out_path = item["out_path"]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(surface, out_path)

            results[orig_idx] = (True, protein_points.shape[0])

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        # Mark all unprocessed as failed
        for orig_idx, item in to_process:
            if results[orig_idx] is None:
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(f"{item['pdb_name']}: {error_msg}\n")
                results[orig_idx] = (False, None)

    return results


def main():
    """Main preprocessing function."""
    import argparse

    enable_timing()

    parser = argparse.ArgumentParser(description="Preprocess PDB files for dMaSIF")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--distance", type=float, default=1.05)
    parser.add_argument("--smoothness", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--nits", type=int, default=4)
    parser.add_argument("--sup_sampling", type=int, default=20)
    parser.add_argument("--variance", type=float, default=0.1)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers for parallel PDB parsing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for GPU processing"
    )

    args = parser.parse_args()

    # Setup paths
    data_dir = os.path.normpath(args.data_dir)
    pdb_dir = os.path.join(data_dir, "pdb")
    out_surf_dir = args.out_dir or os.path.join(data_dir, "surfaces_dmasif")
    log_file = os.path.join(data_dir, "preprocess_dmasif_errors.log")

    if os.path.exists(log_file):
        os.remove(log_file)

    print("dMaSIF preprocessing configuration:")
    print(f"  Data dir:      {data_dir}")
    print(f"  PDB dir:       {pdb_dir}")
    print(f"  Output dir:    {out_surf_dir}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Num workers:   {args.num_workers}")
    print(f"  Recompute:     {args.recompute}")
    print()

    if not os.path.exists(pdb_dir):
        print(f"✘ Error: PDB directory not found: {pdb_dir}")
        return

    # Find PDB files
    pdb_files = [
        os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb")
    ]
    print(f"Found {len(pdb_files)} PDB files\n")

    if not pdb_files:
        print("✘ No PDB files found")
        return

    os.makedirs(out_surf_dir, exist_ok=True)

    # Create dataset and dataloader
    dataset = PDBParsingDataset(pdb_files, out_surf_dir, args.recompute)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Setup GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Process
    t0 = time.time()
    all_results = []

    from tqdm import tqdm

    with time_operation("preprocess_dataset", {"num_proteins": len(dataset)}):
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch_results = process_batch_on_gpu(
                batch,
                device,
                args.distance,
                args.smoothness,
                args.resolution,
                args.nits,
                args.sup_sampling,
                args.variance,
                log_file,
            )
            all_results.extend(batch_results)

    # Statistics
    t1 = time.time()
    point_counts = [r[1] for r in all_results if r[0] and r[1] is not None]
    num_success = sum(1 for r in all_results if r[0])
    num_errors = len(all_results) - num_success

    print("\n✔ Preprocessing complete!")
    print(f"  Surfaces saved to: {out_surf_dir}")
    print(f"  Successful: {num_success}/{len(all_results)}")
    print(f"  Total time: {t1 - t0:.2f} seconds")

    if num_errors > 0:
        print(f"  Failed: {num_errors}")
        print(f"  Errors logged to: {log_file}")

    if point_counts:
        arr = np.array(point_counts)
        print("\n📊 Surface Point Statistics:")
        print(f"  Mean: {np.mean(arr):.1f} ± {np.std(arr):.1f}")
        print(f"  Min: {np.min(arr)}, Max: {np.max(arr)}")

    print_timing_stats()


if __name__ == "__main__":
    main()
