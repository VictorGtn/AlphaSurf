"""
Batched preprocessing script to generate dmasif-compatible surfaces from PDB files.

This script processes PDB files in explicit batches:
1. Select a batch of PDB files.
2. Parse them in parallel (CPU).
3. Generate surfaces in a single batch (GPU).
4. Save results.
"""

import os
import sys
import warnings
from pathlib import Path
import torch
import torch.multiprocessing as mp
import time

# Add project root to path for imports
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "..", "..", ".."))

from alphasurf.protein.surfaces import SurfaceObject
from alphasurf.protein.graphs import parse_pdb_path
from alphasurf.network_utils.misc_arch.dmasif_utils.geometry_processing import (
    atoms_to_points_normals,
)
from alphasurf.network_utils.misc_arch.timing import (
    time_operation,
    enable_timing,
    print_timing_stats,
)


def atom_types_to_onehot(
    atom_types: torch.Tensor, num_atom_types: int = 22
) -> torch.Tensor:
    """
    Convert atom type indices to one-hot encoding.
    """
    N = len(atom_types)
    onehot = torch.zeros(
        N, num_atom_types, dtype=torch.float32, device=atom_types.device
    )

    safe_indices = atom_types.clone()
    safe_indices[safe_indices >= num_atom_types] = num_atom_types - 1

    onehot.scatter_(1, safe_indices.unsqueeze(1), 1.0)

    return onehot


def parse_pdb_wrapper(args):
    """
    Wrapper for parsing PDB to be used in multiprocessing.
    Args: (pdb_path, out_surf_path, recompute)
    Returns: Dict with data or error
    """
    pdb_path, out_surf_path, recompute = args
    pdb_name = Path(pdb_path).stem

    if os.path.exists(out_surf_path) and not recompute:
        return {"skip": True, "pdb_name": pdb_name}

    try:
        # Parse PDB
        arrays = parse_pdb_path(pdb_path, use_pqr=False)
        if arrays is None or arrays[0] is None:
            return {"error": f"Failed to parse {pdb_name}", "pdb_name": pdb_name}

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

        return {
            "atom_pos": atom_pos,  # numpy
            "atom_types": atom_types,  # numpy
            "pdb_name": pdb_name,
            "skip": False,
            "error": None,
        }
    except Exception as e:
        import traceback

        return {"error": f"{str(e)}\n{traceback.format_exc()}", "pdb_name": pdb_name}


def process_batch_gpu(batch_results, args, device, out_surf_dir):
    """
    Process a batch of parsed proteins on GPU.
    """
    # Filter valid
    valid_data = [r for r in batch_results if not r.get("error") and not r.get("skip")]

    if not valid_data:
        return 0  # No proteins processed

    # Collate
    atom_pos_list = [torch.from_numpy(d["atom_pos"]).float() for d in valid_data]
    atom_types_list = [torch.from_numpy(d["atom_types"]).long() for d in valid_data]
    pdb_names = [d["pdb_name"] for d in valid_data]

    # Create batch vector
    batch_vec_list = []
    for i, pos in enumerate(atom_pos_list):
        batch_vec_list.append(torch.full((len(pos),), i, dtype=torch.long))

    # Move to GPU
    atom_pos = torch.cat(atom_pos_list, dim=0).to(device)
    atom_types = torch.cat(atom_types_list, dim=0).to(device)
    batch_atoms = torch.cat(batch_vec_list, dim=0).to(device)

    # One-hot
    atomtypes_onehot = atom_types_to_onehot(atom_types, num_atom_types=22)

    # Geometry processing
    with time_operation(
        "atoms_to_points_normals_batch",
        {"batch_size": len(pdb_names), "total_atoms": len(atom_pos)},
    ):
        points, normals, batch_points = atoms_to_points_normals(
            atom_pos,
            batch_atoms,
            num_atoms=22,
            distance=args.distance,
            smoothness=args.smoothness,
            resolution=args.resolution,
            nits=args.nits,
            atomtypes=atomtypes_onehot,
            sup_sampling=args.sup_sampling,
            variance=args.variance,
        )

    # Save results
    processed_count = 0
    for i, name in enumerate(pdb_names):
        mask = batch_points == i
        if mask.sum() == 0:
            # Handle empty
            p = torch.empty((0, 3), device=device)
            n = torch.empty((0, 3), device=device)
        else:
            p = points[mask]
            n = normals[mask]

        single_batch = torch.zeros(len(p), dtype=torch.long, device=device)

        surface = SurfaceObject(verts=p, faces=None, vnormals=n, batch=single_batch)

        save_path = os.path.join(out_surf_dir, f"{name}.pt")
        torch.save(surface, save_path)
        processed_count += 1

    return processed_count


def main():
    import argparse

    enable_timing()

    parser = argparse.ArgumentParser(
        description="Batched dMaSIF Preprocessing (Explicit Loop)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory containing pdb/ subdirectory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: data_dir/surfaces_dmasif)",
    )
    parser.add_argument(
        "--recompute", action="store_true", help="Recompute even if output exists"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of proteins per GPU batch"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of CPU workers for parsing"
    )

    # dMaSIF parameters
    parser.add_argument("--distance", type=float, default=1.05)
    parser.add_argument("--smoothness", type=float, default=0.5)
    parser.add_argument("--resolution", type=float, default=2.5)
    parser.add_argument("--nits", type=int, default=4)
    parser.add_argument("--sup_sampling", type=int, default=20)
    parser.add_argument("--variance", type=float, default=0.1)

    args = parser.parse_args()

    data_dir = os.path.normpath(args.data_dir)
    pdb_dir = os.path.join(data_dir, "pdb")

    if args.out_dir is None:
        out_surf_dir = os.path.join(data_dir, "surfaces_dmasif")
    else:
        out_surf_dir = os.path.normpath(args.out_dir)

    os.makedirs(out_surf_dir, exist_ok=True)
    log_file = os.path.join(data_dir, "preprocess_dmasif_errors.log")
    if os.path.exists(log_file):
        os.remove(log_file)

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Set start method to spawn for CUDA compatibility/safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(" batched dMaSIF preprocessing (Explicit Loop):")
    print(f"  PDB dir: {pdb_dir}")
    print(f"  Out dir: {out_surf_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Workers: {args.num_workers}")
    print(
        f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )

    if not os.path.exists(pdb_dir):
        print(f"Error: {pdb_dir} not found.")
        return

    pdb_files = [
        os.path.join(pdb_dir, f) for f in os.listdir(pdb_dir) if f.endswith(".pdb")
    ]
    # Sort for deterministic order
    pdb_files.sort()

    print(f"Found {len(pdb_files)} PDB files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_success = 0
    total_failed = 0
    t0 = time.time()

    # Pre-calculate arguments for all files to be safe/simple
    # or iterate in chunks

    # We use a persistent pool if workers > 0
    pool = mp.Pool(processes=args.num_workers) if args.num_workers > 0 else None

    try:
        # Loop over batches
        for i in range(0, len(pdb_files), args.batch_size):
            batch_files = pdb_files[i : i + args.batch_size]

            # Prepare args for map
            map_args = [
                (f, os.path.join(out_surf_dir, Path(f).stem + ".pt"), args.recompute)
                for f in batch_files
            ]

            # 1. Parse (CPU)
            if pool:
                batch_results = pool.map(parse_pdb_wrapper, map_args)
            else:
                batch_results = [parse_pdb_wrapper(a) for a in map_args]

            # Log errors
            current_errors = [r for r in batch_results if r.get("error")]
            if current_errors:
                with open(log_file, "a") as log:
                    for err in current_errors:
                        log.write(f"{err['pdb_name']}: {err['error']}\n")
                total_failed += len(current_errors)

            # 2. Process (GPU)
            # Only process if we have valid data
            with time_operation("process_batch_gpu", {"batch_size": len(batch_files)}):
                processed = process_batch_gpu(batch_results, args, device, out_surf_dir)
            total_success += processed

            # Optional: Clear GPU cache if memory is tight
            # torch.cuda.empty_cache()

            print(
                f"Processed batch {i//args.batch_size + 1}/{(len(pdb_files)+args.batch_size-1)//args.batch_size}: {processed} surfaces generated."
            )

    finally:
        if pool:
            pool.close()
            pool.join()

    print(f"\nProcessing complete in {time.time() - t0:.2f}s")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_failed}")
    if total_failed > 0:
        print(f"Errors logged to {log_file}")
    print_timing_stats()


if __name__ == "__main__":
    main()
