"""
Script to precompute surfaces and graphs for PINDER data.
Allows switching training to 'disk' mode for massive speedup.

Usage:
    python tasks/pinder_pair/precompute.py
"""

import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path

import hydra
import torch
from tqdm import tqdm

# Add project root to path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.pinder_pair.dataset import load_pinder_split


def get_loader(cfg):
    """Initialize ProteinLoader from config."""
    # Force on_fly mode
    from omegaconf import OmegaConf, open_dict

    # Merge on_fly settings into surface config to ensure params like
    # surface_method, alpha_value, etc. are passed to ProteinLoader
    with open_dict(cfg.cfg_surface):
        surface_cfg = OmegaConf.merge(cfg.cfg_surface, cfg.on_fly)

    return ProteinLoader(
        mode="on_fly",
        pdb_dir=os.path.join(cfg.data_dir, "pdb"),
        surface_config=surface_cfg,
        graph_config=cfg.cfg_graph,
        noise_augmentor=None,
    )


def process_protein(name, cfg, surface_dir, graph_dir, recompute=False):
    """Process a single protein and save results."""
    # Check if already done
    surf_path = os.path.join(surface_dir, f"{name}.pt")
    graph_path = os.path.join(graph_dir, f"{name}.pt")

    if not recompute and os.path.exists(surf_path) and os.path.exists(graph_path):
        return "skipped", 0.0, 0.0, 0.0

    try:
        # Initialize loader locally to avoid pickling issues with CGAL bindings
        loader = get_loader(cfg)

        # Load (generates on fly)
        protein = loader.load(name)

        if protein is None:
            return f"failed: {name} load returned None", 0.0, 0.0, 0.0

        drop_ratio = 0.0
        singular_edges = 0.0
        singular_faces = 0.0
        # Save Surface
        if protein.surface is not None:
            # We save the SurfaceObject directly
            # Note: We strip features to save space?
            # ProteinLoader.load_from_disk expects '.pt' with torch.load
            torch.save(protein.surface, surf_path)
            if hasattr(protein.surface, "drop_ratio"):
                drop_ratio = protein.surface.drop_ratio
            if hasattr(protein.surface, "singular_edges"):
                singular_edges = protein.surface.singular_edges
            if hasattr(protein.surface, "singular_faces"):
                singular_faces = protein.surface.singular_faces

        # Save Graph
        if protein.graph is not None:
            torch.save(protein.graph, graph_path)

        return "success", drop_ratio, singular_edges, singular_faces

    except Exception as e:
        return f"error: {name} {str(e)}", 0.0, 0.0, 0.0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    print(f"Precomputing PINDER data using config: {cfg.on_fly.surface_method}")

    # Setup Output Dirs
    # We use a subfolder based on surface method to avoid collisions
    method_str = f"{cfg.on_fly.surface_method}_{cfg.on_fly.face_reduction_rate}"
    if cfg.on_fly.surface_method == "alpha_complex":
        method_str += f"_a{cfg.on_fly.alpha_value}"

    surface_dir = os.path.join(cfg.data_dir, "surfaces", method_str)
    graph_dir = os.path.join(cfg.data_dir, "graphs", method_str)

    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    print(f"Output Surfaces: {surface_dir}")
    print(f"Output Graphs:   {graph_dir}")

    recompute = False
    if "preprocessing" in cfg and "recompute_surfaces" in cfg.preprocessing:
        recompute = cfg.preprocessing.recompute_surfaces

    print(f"Recompute Surfaces: {recompute}")

    # Collect all PDB IDs
    all_names = set()

    # 1. Train/Val (Holo)
    for split in ["train", "val"]:
        systems = load_pinder_split(cfg.data_dir, split)
        print(f"Loaded {len(systems)} systems from {split}")
        for s in systems:
            all_names.add(s["receptor_id"])
            all_names.add(s["ligand_id"])

    # 2. Test (All settings)
    # Even if config says 'apo', we try to generate all provided csvs
    # Or just iterate keys in data_dir
    csv_files = list(Path(cfg.data_dir).glob("systems_test_*.csv"))
    for csv_path in csv_files:
        setting = csv_path.stem.replace("systems_test_", "")
        systems = load_pinder_split(cfg.data_dir, "test", setting)
        print(f"Loaded {len(systems)} systems from test_{setting}")
        for s in systems:
            all_names.add(s["receptor_id"])
            all_names.add(s["ligand_id"])

    names_list = sorted(list(all_names))
    print(f"Total unique proteins to process: {len(names_list)}")

    # Processing
    debug_mode = False

    if debug_mode:
        # Serial
        results = []
        for name in tqdm(names_list):
            results.append(
                process_protein(name, cfg, surface_dir, graph_dir, recompute=recompute)
            )
    else:
        # Parallel
        num_workers = cfg.loader.num_workers
        print(f"Starting {num_workers} workers...")

        func = partial(
            process_protein,
            cfg=cfg,
            surface_dir=surface_dir,
            graph_dir=graph_dir,
            recompute=recompute,
        )

        # Use spawn to potential avoid CGAL/Torch issues
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            results = list(
                tqdm(pool.imap_unordered(func, names_list), total=len(names_list))
            )

    # Summary
    success = [r for r, _, _, _ in results if r == "success"]
    skipped = [r for r, _, _, _ in results if r == "skipped"]
    errors = [
        r for r, _, _, _ in results if r.startswith("error") or r.startswith("failed")
    ]
    drop_ratios = [d for r, d, _, _ in results if r == "success"]
    singular_edges_list = [s for r, _, s, _ in results if r == "success"]
    singular_faces_list = [f for r, _, _, f in results if r == "success"]

    print("\nDone.")
    print(f"Success: {len(success)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors:  {len(errors)}")

    if len(errors) > 0:
        print("\nError Details (First 20 unique):")
        unique_errors = sorted(list(set(errors)))[:20]
        for err in unique_errors:
            print(f"  {err}")

    if len(drop_ratios) > 0:
        import numpy as np

        drop_ratios = np.array(drop_ratios)
        print("\nComponent Drop Statistics:")
        for threshold in [0.1, 0.2, 0.3, 0.4]:
            count = np.sum(drop_ratios > threshold)
            pct = (count / len(drop_ratios)) * 100
            print(f"  > {threshold:.0%} faces dropped: {count} proteins ({pct:.1f}%)")

    if len(singular_edges_list) > 0:
        singular_edges_arr = np.array(singular_edges_list)
        print("\nDropped Singular/Regular Edges (Wires):")
        print(f"  Total dropped: {np.sum(singular_edges_arr):.0f}")
        print(f"  Mean per protein: {np.mean(singular_edges_arr):.1f}")
        print(f"  Max per protein: {np.max(singular_edges_arr):.0f}")
        print(f"  > 0 edges dropped: {np.sum(singular_edges_arr > 0)} proteins")

    if len(singular_faces_list) > 0:
        singular_faces_arr = np.array(singular_faces_list)
        print("\nSingular Faces Statistics:")
        print(f"  Total singular faces: {np.sum(singular_faces_arr):.0f}")
        print(f"  Mean per protein: {np.mean(singular_faces_arr):.1f}")
        print(f"  Max per protein: {np.max(singular_faces_arr):.0f}")
        print(f"  > 0 singular faces: {np.sum(singular_faces_arr > 0)} proteins")

    print("\nTo use this data, update your training config:")
    print("on_fly: null")
    print(f"surface_dir: {surface_dir}")
    print(f"graph_dir: {graph_dir}")


if __name__ == "__main__":
    main()
