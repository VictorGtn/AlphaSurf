import argparse
import glob
import os
import sys
import time

import numpy as np
import open3d as o3d
from tqdm import tqdm

# Add project root to path
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, ".."))

import torch
from alphasurf.protein.create_operators import compute_operators


def benchmark_mesh(file_path):
    print(f"Processing: {os.path.basename(file_path)}")

    # Load mesh
    try:
        if file_path.endswith(".ply"):
            mesh = o3d.io.read_triangle_mesh(file_path)
            verts = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.triangles, dtype=np.int32)
        elif file_path.endswith(".pt"):
            data = torch.load(file_path, map_location="cpu")

            verts = None
            faces = None

            if isinstance(data, dict):
                verts = data.get("verts")
                if verts is None:
                    verts = data.get("vertices")
                faces = data.get("faces")
                if faces is None:
                    faces = data.get("triangles")
            else:
                # PyGMO or similar object
                if hasattr(data, "verts"):
                    verts = data.verts
                elif hasattr(data, "vertices"):
                    verts = data.vertices

                if hasattr(data, "faces"):
                    faces = data.faces
                elif hasattr(data, "triangles"):
                    faces = data.triangles

            if isinstance(verts, torch.Tensor):
                verts = verts.numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.numpy()

            if verts is None or faces is None:
                print("  Skipping: Could not find verts/faces in .pt file")
                return None

            verts = verts.astype(np.float32)
            faces = faces.astype(np.int32)
        else:
            print(
                f"  Skipping: Unsupported file extension {os.path.splitext(file_path)[1]}"
            )
            return None

        n_verts = len(verts)
        print(f"  Vertices: {n_verts}, Faces: {len(faces)}")

        if n_verts < 5:
            print("  Skipping: Too few vertices")
            return None

        # Benchmark Standard
        start_time = time.time()
        try:
            compute_operators(verts, faces, k_eig=128, use_robust_laplacian=False)
            time_std = time.time() - start_time
            print(f"  Standard Laplacian: {time_std:.4f}s")
        except Exception as e:
            print(f"  Standard Failed: {e}")
            time_std = None

        # Benchmark Robust
        start_time = time.time()
        try:
            compute_operators(verts, faces, k_eig=128, use_robust_laplacian=True)
            time_rob = time.time() - start_time
            print(f"  Robust Laplacian:   {time_rob:.4f}s")
        except Exception as e:
            print(f"  Robust Failed: {e}")
            time_rob = None

        return {
            "name": os.path.basename(file_path),
            "n_verts": n_verts,
            "time_std": time_std,
            "time_rob": time_rob,
        }

    except Exception as e:
        print(f"Error loading mesh: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Standard vs Robust Laplacian"
    )
    parser.add_argument(
        "input_dir", type=str, help="Directory containing mesh files (.ply or .pt)"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Max number of files to process"
    )
    args = parser.parse_args()

    # Search for both .ply and .pt
    files = glob.glob(os.path.join(args.input_dir, "*.ply")) + glob.glob(
        os.path.join(args.input_dir, "*.pt")
    )
    if not files:
        print(f"No mesh files (.ply or .pt) found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} files. Processing first {args.limit}...")
    files = files[: args.limit]

    results = []
    for f in tqdm(files):
        res = benchmark_mesh(f)
        if res:
            results.append(res)

    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY")
    print("=" * 50)

    valid_results = [
        r for r in results if r["time_std"] is not None and r["time_rob"] is not None
    ]
    if not valid_results:
        print("No valid comparisons made.")
        return

    avg_std = np.mean([r["time_std"] for r in valid_results])
    avg_rob = np.mean([r["time_rob"] for r in valid_results])

    print(f"Processed {len(valid_results)} meshes successfully.")
    print(f"Average Time (Standard): {avg_std:.4f}s")
    print(f"Average Time (Robust):   {avg_rob:.4f}s")
    print(f"Ratio (Robust/Standard): {avg_rob / avg_std:.2f}x")


if __name__ == "__main__":
    main()
