#!/usr/bin/env python3
"""Measure EDTSurf patch vertex counts at the patch-extraction radii."""

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPEN3D_CPU_THREAD_COUNT", "1")

import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def largest_component_count(mask, verts, faces):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return 0

    old_to_new = np.full(len(verts), -1, dtype=np.int64)
    old_to_new[indices] = np.arange(len(indices))
    face_mask = np.all(mask[faces], axis=1)
    masked_faces = faces[face_mask]
    if len(masked_faces) == 0:
        return int(len(indices))

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts[indices]),
        o3d.utility.Vector3iVector(old_to_new[masked_faces]),
    )
    clusters, triangle_counts, _ = mesh.cluster_connected_triangles()
    triangle_counts = np.asarray(triangle_counts)
    if len(triangle_counts) == 0:
        return int(len(indices))

    largest = int(np.argmax(triangle_counts))
    cluster_faces = np.asarray(clusters) == largest
    return int(np.unique(np.asarray(old_to_new[masked_faces[cluster_faces]])).size)


def radius_for_target(distances, verts, faces, target, start, stop, step):
    radius = start
    last_count = 0
    while radius <= stop + 1e-6:
        last_count = largest_component_count(distances <= radius, verts, faces)
        if last_count >= target:
            return float(radius), last_count
        radius += step
    return None, last_count


def measure_one(task):
    (
        pocket,
        split,
        pdb_dir,
        patch_dir,
        grid_scale,
        radii,
        target_verts,
        start_radius,
        max_radius,
        radius_step,
    ) = task
    patch_path = patch_dir / f"{pocket}.npz"
    result = {"pocket": pocket, "split": split, "grid_scale": grid_scale}
    if not patch_path.exists():
        result["status"] = "missing_patch"
        return result

    pdb_path = pdb_dir / f"{pocket.split('_patch_')[0]}.pdb"
    try:
        from alphasurf.protein.create_surface import pdb_to_edtsurf

        with np.load(patch_path, allow_pickle=True) as patch_data:
            ref_verts = np.asarray(patch_data["pkt_verts"], dtype=np.float32)
        verts, faces = pdb_to_edtsurf(str(pdb_path), grid_scale=grid_scale)
        distances = cKDTree(ref_verts).query(verts, k=1)[0]
        result["status"] = "ok"
        result["surface_vertices"] = int(len(verts))
        result["surface_faces"] = int(len(faces))
        for radius in radii:
            mask = distances <= radius
            result[f"selected_{radius:g}"] = int(mask.sum())
            result[f"largest_cc_{radius:g}"] = largest_component_count(
                mask, verts, faces
            )
        if target_verts is not None:
            required_radius, count = radius_for_target(
                distances,
                verts,
                faces,
                target_verts,
                start_radius,
                max_radius,
                radius_step,
            )
            result["required_radius"] = required_radius
            result["largest_cc_at_required_radius"] = count
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)[:240]
        return result


def load_tasks(data_dir):
    splits_dir = data_dir / "raw_data_MasifLigand" / "splits"
    tasks = []
    for split in ("train", "val", "test"):
        with open(splits_dir / f"{split}.p", "rb") as handle:
            pockets = pickle.load(handle)
        tasks.extend((pocket, split) for pocket in pockets)
    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--grid-scale", type=float, default=0.4)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-pockets", type=int)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--target-verts", type=int)
    parser.add_argument("--start-radius", type=float, default=6.0)
    parser.add_argument("--max-radius", type=float, default=12.0)
    parser.add_argument("--radius-step", type=float, default=2.0)
    args = parser.parse_args()

    data_dir = args.data_dir
    tasks = load_tasks(data_dir)
    if args.max_pockets is not None:
        tasks = tasks[: args.max_pockets]

    pdb_dir = data_dir / "raw_data_MasifLigand" / "pdb"
    patch_dir = data_dir / "dataset_MasifLigand"
    radii = (6.0, 8.0, 10.0, 12.0)
    work = [
        (
            pocket,
            split,
            pdb_dir,
            patch_dir,
            args.grid_scale,
            radii,
            args.target_verts,
            args.start_radius,
            args.max_radius,
            args.radius_step,
        )
        for pocket, split in tasks
    ]

    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(args.workers) as pool:
        for index, row in enumerate(pool.imap_unordered(measure_one, work), 1):
            rows.append(row)
            if index % 100 == 0 or index == len(work):
                print(f"{index}/{len(work)}", flush=True)

    rows.sort(key=lambda row: (row["split"], row["pocket"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pocket",
        "split",
        "grid_scale",
        "status",
        "surface_vertices",
        "surface_faces",
        "selected_6",
        "largest_cc_6",
        "selected_8",
        "largest_cc_8",
        "selected_10",
        "largest_cc_10",
        "selected_12",
        "largest_cc_12",
        "error",
    ]
    if args.target_verts is not None:
        fields.insert(-1, "required_radius")
        fields.insert(-1, "largest_cc_at_required_radius")
    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [row for row in rows if row["status"] == "ok"]
    counts = [row["largest_cc_12"] for row in ok]
    print(f"wrote {args.output_csv} ({len(rows)} pockets, {len(ok)} surfaces)")
    if counts:
        print(f"12 A largest-component vertices: min={min(counts)}")
    if args.target_verts is not None:
        required = [row["required_radius"] for row in ok if row["required_radius"]]
        missing = len(ok) - len(required)
        print(
            f"{args.target_verts} vertices: max radius={max(required) if required else 'none'}"
            f"; not reached by {args.max_radius:g} A: {missing}"
        )


if __name__ == "__main__":
    main()
