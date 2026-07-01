"""
Script to precompute surfaces and graphs for PINDER data.
Allows switching training to 'disk' mode for massive speedup.

Usage:
    python tasks/pinder_pair/precompute.py
"""

import multiprocessing
import os
import queue
import sys
import threading
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.pinder_pair.dataset import load_pinder_split


def _process_task(task):
    save_name, pdb_path, cfg, surface_dir, graph_dir, recompute, dry_run = task
    return process_protein(
        save_name,
        cfg,
        surface_dir,
        graph_dir,
        recompute=recompute,
        pdb_path=pdb_path,
        dry_run=dry_run,
    )


def get_loader(cfg):
    """Initialize ProteinLoader from config."""
    # Force on_fly mode
    from omegaconf import OmegaConf, open_dict

    # Merge on_fly settings into surface config to ensure params like
    # surface_method, alpha_value, etc. are passed to ProteinLoader
    with open_dict(cfg.cfg_surface):
        surface_cfg = OmegaConf.merge(cfg.cfg_surface, cfg.on_fly)
        surface_cfg.use_poisson = "poisson" in cfg.encoder.name

    return ProteinLoader(
        mode="on_fly",
        pdb_dir=os.path.join(cfg.data_dir, "pdb"),
        surface_config=surface_cfg,
        graph_config=cfg.cfg_graph,
        noise_augmentor=None,
    )


def process_protein(
    name, cfg, surface_dir, graph_dir, recompute=False, pdb_path=None, dry_run=False
):
    """Process a single protein and save results."""
    surf_path = os.path.join(surface_dir, f"{name}.pt")
    graph_path = os.path.join(graph_dir, f"{name}.pt")

    if (
        not dry_run
        and not recompute
        and os.path.exists(surf_path)
        and os.path.exists(graph_path)
    ):
        return "skipped", 0.0, 0.0, 0.0, 0.0, 0.0

    t_start = time.time()
    try:
        loader = get_loader(cfg)
        protein = loader.load(name, pdb_path=pdb_path)
        t_compute = time.time() - t_start

        if protein is None:
            return f"failed: {name} load returned None", 0.0, 0.0, 0.0, 0.0, t_compute

        drop_ratio = 0.0
        drop_ratio_vertex = 0.0
        singular_edges = 0.0
        singular_faces = 0.0

        if protein.surface is None:
            return f"failed: {name} no surface", 0.0, 0.0, 0.0, 0.0, t_compute

        if hasattr(protein.surface, "drop_ratio"):
            drop_ratio = protein.surface.drop_ratio
        if hasattr(protein.surface, "drop_ratio_vertex"):
            drop_ratio_vertex = protein.surface.drop_ratio_vertex
        if hasattr(protein.surface, "singular_edges"):
            singular_edges = protein.surface.singular_edges
        if hasattr(protein.surface, "singular_faces"):
            singular_faces = protein.surface.singular_faces

        if protein.graph is None:
            return f"failed: {name} no graph", 0.0, 0.0, 0.0, 0.0, t_compute

        if not dry_run:
            torch.save(protein.surface, surf_path)
            if protein.metadata.get("atom_pos") is not None:
                protein.graph.atom_pos = protein.metadata["atom_pos"]
            if protein.metadata.get("atom_res_map") is not None:
                protein.graph.atom_res_map = protein.metadata["atom_res_map"]
            torch.save(protein.graph, graph_path)

        return (
            "success",
            drop_ratio,
            drop_ratio_vertex,
            singular_edges,
            singular_faces,
            t_compute,
        )

    except Exception as e:
        return f"error: {name} {str(e)}", 0.0, 0.0, 0.0, 0.0, time.time() - t_start


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    print(f"Precomputing PINDER data using config: {cfg.on_fly.surface_method}")

    # Setup Output Dirs
    # We use a subfolder based on surface method to avoid collisions
    method_str = f"{cfg.on_fly.surface_method}_{cfg.on_fly.face_reduction_rate}"
    if cfg.on_fly.surface_method == "alpha_complex":
        method_str += f"_a{cfg.on_fly.alpha_value}"
    elif cfg.on_fly.surface_method in ("edtsurf", "nanoshaper"):
        if cfg.on_fly.surface_method == "edtsurf":
            gs = cfg.on_fly.get("edtsurf_grid_scale")
        else:
            gs = cfg.on_fly.get("nanoshaper_grid_scale")
        if gs is not None:
            method_str += f"_gs{gs}"

    surface_dir = os.path.join(cfg.data_dir, "surfaces", method_str)
    graph_dir = os.path.join(cfg.data_dir, "graphs", method_str)

    os.makedirs(surface_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    print(f"Output Surfaces: {surface_dir}")
    print(f"Output Graphs:   {graph_dir}")

    recompute = False
    if "preprocessing" in cfg and "recompute_surfaces" in cfg.preprocessing:
        recompute = cfg.preprocessing.recompute_surfaces

    dry_run = False
    if "preprocessing" in cfg and "dry_run" in cfg.preprocessing:
        dry_run = cfg.preprocessing.dry_run

    print(f"Recompute Surfaces: {recompute}")
    print(f"Dry run: {dry_run}")

    # Collect all proteins to process as (save_name, pdb_path) tuples.
    # Test systems get a setting suffix to match PDB naming (_holo, _apo, _af2).
    pdb_dir_path = os.path.join(cfg.data_dir, "pdb")
    all_tasks = []

    # 1. Train/Val (Holo) — save with _holo suffix, PDB has no suffix
    for split in ["train", "val"]:
        systems = load_pinder_split(cfg.data_dir, split)
        print(f"Loaded {len(systems)} systems from {split}")
        for s in systems:
            for key in ("receptor_id", "ligand_id"):
                pdb_path = os.path.join(pdb_dir_path, f"{s[key]}.pdb")
                all_tasks.append((f"{s[key]}_holo", pdb_path))

    # 2. Test (All settings) — save with setting suffix
    csv_files = list(Path(cfg.data_dir).glob("systems_test_*.csv"))
    for csv_path in csv_files:
        setting = csv_path.stem.replace("systems_test_", "")
        systems = load_pinder_split(cfg.data_dir, "test", setting)
        print(f"Loaded {len(systems)} systems from test_{setting}")
        for s in systems:
            for key in ("receptor_id", "ligand_id"):
                name = s[key]
                save_name = f"{name}_{setting}"
                setting_pdb = os.path.join(pdb_dir_path, f"{name}_{setting}.pdb")
                all_tasks.append((save_name, setting_pdb))

    # Deduplicate by save_name (keep first occurrence, train/val before test)
    seen = set()
    unique_tasks = []
    for save_name, pdb in all_tasks:
        if save_name not in seen:
            seen.add(save_name)
            unique_tasks.append((save_name, pdb))

    print(f"Total unique proteins to process: {len(unique_tasks)}")

    # Processing
    t_wall_start = time.time()

    debug_mode = False

    if debug_mode:
        results = []
        for save_name, pdb_path in tqdm(unique_tasks):
            results.append(
                process_protein(
                    save_name,
                    cfg,
                    surface_dir,
                    graph_dir,
                    recompute=recompute,
                    pdb_path=pdb_path,
                )
            )
    else:
        num_workers = cfg.loader.num_workers
        print(f"Starting {num_workers} workers...")

        packed_tasks = [
            (sn, pp, cfg, surface_dir, graph_dir, recompute, dry_run)
            for sn, pp in unique_tasks
        ]

        STALL_TIMEOUT = 600
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(num_workers)

        def _collect(q, it):
            try:
                for r in it:
                    q.put(("result", r))
            except Exception as e:
                q.put(("error", str(e)))

        results = []
        pbar = tqdm(total=len(unique_tasks))
        q = queue.Queue()
        collector = threading.Thread(
            target=_collect,
            args=(q, pool.imap_unordered(_process_task, packed_tasks)),
            daemon=True,
        )
        collector.start()

        last_progress = time.time()
        while True:
            try:
                tag, val = q.get(timeout=2)
            except queue.Empty:
                if time.time() - last_progress > STALL_TIMEOUT:
                    print(
                        f"\nStall detected: no result for {STALL_TIMEOUT}s. Aborting."
                    )
                    break
                continue

            if tag == "error":
                print(f"\nPool error: {val}")
                break

            results.append(val)
            pbar.update(1)
            last_progress = time.time()

            if pbar.n >= len(unique_tasks):
                break

        pbar.close()
        pool.terminate()
        pool.join()

    t_wall = time.time() - t_wall_start

    # Summary
    success = [r for r, _, _, _, _, _ in results if r == "success"]
    skipped = [r for r, _, _, _, _, _ in results if r == "skipped"]
    errors = [
        r
        for r, _, _, _, _, _ in results
        if r.startswith("error") or r.startswith("failed")
    ]
    drop_ratios = [d for r, d, _, _, _, _ in results if r == "success"]
    drop_ratios_vertex = [d for r, _, d, _, _, _ in results if r == "success"]
    singular_edges_list = [s for r, _, _, s, _, _ in results if r == "success"]
    singular_faces_list = [f for r, _, _, _, f, _ in results if r == "success"]
    proc_times = [t for r, _, _, _, _, t in results if r == "success"]

    print("\nDone.")
    print(f"Success: {len(success)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Errors:  {len(errors)}")
    print(f"Wall clock: {t_wall:.1f}s ({t_wall / 60:.1f} min)")
    if len(proc_times) > 0:
        proc_arr = np.array(proc_times)
        print(
            f"Avg per protein: {np.mean(proc_arr):.3f}s  "
            f"(median {np.median(proc_arr):.3f}s, max {np.max(proc_arr):.3f}s)"
        )
        print(f"Throughput: {len(success) / t_wall:.2f} proteins/s")

    if len(errors) > 0:
        print("\nError Details (First 20 unique):")
        unique_errors = sorted(list(set(errors)))[:20]
        for err in unique_errors:
            print(f"  {err}")

    if len(drop_ratios) > 0:
        drop_ratios = np.array(drop_ratios)
        print("\nComponent Drop Statistics (edge-sharing):")
        for threshold in [0.1, 0.2, 0.3, 0.4]:
            count = np.sum(drop_ratios > threshold)
            pct = (count / len(drop_ratios)) * 100
            print(f"  > {threshold:.0%} faces dropped: {count} proteins ({pct:.1f}%)")

    if len(drop_ratios_vertex) > 0:
        drop_ratios_vertex = np.array(drop_ratios_vertex)
        n_nonzero = np.sum(drop_ratios_vertex > 0)
        print(
            "\nComponent Drop Statistics (vertex-sharing, same baseline as edge-based):"
        )
        print(
            f"  Proteins with any vertex-only components: {n_nonzero} ({n_nonzero / len(drop_ratios_vertex) * 100:.1f}%)"
        )
        for threshold in [0.1, 0.2, 0.3, 0.4]:
            count = np.sum(drop_ratios_vertex > threshold)
            pct = (count / len(drop_ratios_vertex)) * 100
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
