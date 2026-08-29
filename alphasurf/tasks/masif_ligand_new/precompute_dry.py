import argparse
import csv
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from alphasurf.tasks.masif_ligand_new.dataset import load_ligand_data
from alphasurf.tasks.pinder_pair.precompute import _process_task
from alphasurf.utils.timing_stats import print_summary, reset


def build_config(args):
    use_whole_surfaces = args.patch_dir is None
    return OmegaConf.create(
        {
            "data_dir": str(args.pdb_dir.parent),
            "encoder": {"name": "pronet_gvpencoder"},
            "on_fly": {
                "surface_method": args.surface_method,
                "alpha_value": args.alpha_value,
                "face_reduction_rate": args.face_reduction_rate,
                "max_vert_number": 100000,
                "min_vert_number": 16,
                "use_pymesh": False,
                "use_whole_surfaces": use_whole_surfaces,
                "reference_patch_dir": (
                    str(args.patch_dir) if args.patch_dir is not None else None
                ),
                "patch_radius": 6.0,
                "min_verts": args.min_verts,
                "patch_max_radius": args.patch_max_radius,
                "nanoshaper_grid_scale": args.grid_scale,
                "edtsurf_grid_scale": args.grid_scale,
                "use_igl_normals": False,
                "tufting": True,
            },
            "cfg_surface": {
                "use_surfaces": True,
                "use_whole_surfaces": use_whole_surfaces,
                "feat_keys": "all",
                "oh_keys": "all",
            },
            "cfg_graph": {
                "use_graphs": True,
                "feat_keys": "all",
                "oh_keys": "all",
                "use_esm": False,
                "read_b_factors": False,
            },
        }
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path)
    parser.add_argument(
        "--surface-method",
        choices=("alpha_complex", "edtsurf", "msms", "nanoshaper"),
        required=True,
    )
    parser.add_argument("--grid-scale", type=float, default=0.5)
    parser.add_argument("--alpha-value", type=float, default=0.0)
    parser.add_argument("--face-reduction-rate", type=float, default=1.0)
    parser.add_argument("--min-verts", type=int, default=140)
    parser.add_argument("--patch-max-radius", type=float, default=12.0)
    parser.add_argument(
        "--no-patch-radius-cap",
        dest="patch_max_radius",
        action="store_const",
        const=None,
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-proteins", type=int)
    parser.add_argument("--stage-timings", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    missing_patches = []
    if args.patch_dir is None:
        paths = sorted(args.pdb_dir.glob("*.pdb"))
        if args.max_proteins is not None:
            paths = paths[: args.max_proteins]
        samples = [(path.stem, path) for path in paths]
    else:
        raw_dir = args.pdb_dir.parent
        expected = {}
        for split in ("train", "val", "test"):
            expected.update(
                load_ligand_data(
                    str(raw_dir / "splits" / f"{split}-list.txt"),
                    str(raw_dir / "ligand"),
                    str(raw_dir / "splits" / f"{split}.p"),
                )
            )
        names = sorted(expected)
        if args.max_proteins is not None:
            names = names[: args.max_proteins]
        present = {path.stem for path in args.patch_dir.glob("*.npz")}
        missing_patches = [name for name in names if name not in present]
        samples = [
            (
                name,
                args.pdb_dir / f"{name.split('_patch_')[0]}.pdb",
            )
            for name in names
            if name in present
        ]
    cfg = build_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ALPHASURF_TIMING_ENABLED"] = "1" if args.stage_timings else "0"

    tasks = [
        (
            name,
            str(path),
            cfg,
            str(args.output_dir / "surfaces"),
            str(args.output_dir / "graphs"),
            True,
            True,
        )
        for name, path in samples
    ]

    print(f"Surface method: {args.surface_method}")
    print(f"Grid scale: {args.grid_scale}")
    print(f"Face reduction rate: {args.face_reduction_rate}")
    print(f"Workers: {args.workers}")
    print(f"PDB directory: {args.pdb_dir}")
    if args.patch_dir is not None:
        print(f"Patch directory: {args.patch_dir}")
    print(f"Total expected samples: {len(tasks) + len(missing_patches)}")
    print(f"Available patches to process: {len(tasks)}")
    print(f"Missing patches: {len(missing_patches)}")

    if args.stage_timings:
        reset()
    start = time.time()
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(_process_task, tasks), total=len(tasks)))
        pool.close()
        pool.join()
    wall_time = time.time() - start
    if args.stage_timings:
        print_summary()

    rows = []
    for (name, _), result in zip(samples, results):
        (
            status,
            drop_ratio,
            drop_ratio_vertex,
            singular_edges,
            singular_faces,
            elapsed,
        ) = result
        rows.append(
            {
                "pdb": name,
                "status": status,
                "elapsed": elapsed,
                "drop_ratio": drop_ratio,
                "drop_ratio_vertex": drop_ratio_vertex,
                "singular_edges": singular_edges,
                "singular_faces": singular_faces,
            }
        )
    rows.extend(
        {
            "pdb": name,
            "status": f"failed: {name} patch file missing",
            "elapsed": 0.0,
            "drop_ratio": 0.0,
            "drop_ratio_vertex": 0.0,
            "singular_edges": 0.0,
            "singular_faces": 0.0,
        }
        for name in missing_patches
    )

    output_csv = args.output_dir / "full_pipeline_raw.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["status"] == "success"]
    failures = [row for row in rows if row["status"] != "success"]
    elapsed = np.asarray([row["elapsed"] for row in successful])

    print(f"Success: {len(successful)}")
    print(f"Errors: {len(failures)}")
    print(f"Wall clock: {wall_time:.1f}s ({wall_time / 60:.1f} min)")
    if len(elapsed):
        print(
            f"Mean worker task time: {elapsed.mean():.3f}s "
            f"(median {np.median(elapsed):.3f}s, max {elapsed.max():.3f}s)"
        )
        print(f"Throughput: {len(successful) / wall_time:.2f} proteins/s")
    for row in failures[:20]:
        print(row["status"])


if __name__ == "__main__":
    main()
