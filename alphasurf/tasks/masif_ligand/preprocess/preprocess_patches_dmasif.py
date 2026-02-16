"""
Preprocessing script to create subsetted dmasif surfaces from patches.

For each .npz patch file:
1. Load patch vertices (defines the region of interest)
2. Load full dmasif surface (point cloud, no mesh)
3. Find all surface points within radius of any patch vertex
4. Optionally extract largest connected component using point cloud clustering
5. Save subsetted surface

Note: dmasif surfaces are point clouds (no faces), so we use point-based clustering
instead of mesh-based connected component extraction.
"""

import os
import sys
import warnings
from pathlib import Path
import numpy as np
import torch
import multiprocessing
from typing import Dict, Tuple, Optional

try:
    import open3d as o3d

    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    warnings.warn("Open3D not available, will skip connected component extraction")

# Suppress warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Set the start method for multiprocessing to 'spawn' for CUDA compatibility
    # This must be done at the beginning of the main execution block
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # This will be raised if the start method has already been set.
        # We can safely ignore this if we know it has been set to 'spawn'.
        pass

    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, "..", "..", ".."))

from alphasurf.utils.python_utils import do_all_simple
from alphasurf.protein.surfaces import SurfaceObject


def load_patch(patch_path: str) -> Dict:
    """Load patch vertices and faces from npz file."""
    data = np.load(patch_path, allow_pickle=True)
    return {
        "verts": data["pkt_verts"],
        "faces": data["pkt_faces"].astype(int) if "pkt_faces" in data else None,
        "patch_name": os.path.basename(patch_path).replace(".npz", ""),
    }


def load_surface(surface_path: str) -> SurfaceObject:
    """Load full dmasif surface from .pt file."""
    # Load on GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.load(surface_path, weights_only=False, map_location=device)


def compute_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between two sets of points.
    Returns array of shape (len(points_a),) with minimum distance to points_b.
    """
    # Vectorized computation for efficiency
    # points_a: (N, 3), points_b: (M, 3)
    # Returns: (N,) minimum distances
    diff = points_a[:, None, :] - points_b[None, :, :]  # (N, M, 3)
    dists = np.linalg.norm(diff, axis=2)  # (N, M)
    min_dists = np.min(dists, axis=1)  # (N,)
    return min_dists


def find_points_near_patch(
    surface_verts: np.ndarray,
    patch_verts: np.ndarray,
    radius: float = 6.0,
    min_points: int = 50,
    max_radius: float = 20.0,
) -> np.ndarray:
    """
    Find all surface points within radius Angstroms of any patch vertex.
    Adaptively increases radius to ensure at least min_points are selected.

    Returns: boolean mask of shape (n_surface_verts,)
    """
    # Compute minimum distance from each surface point to patch
    min_dists = compute_distances(surface_verts, patch_verts)

    # Start with initial radius
    current_radius = radius
    point_mask = min_dists <= current_radius

    # Adaptively increase radius if needed to meet minimum point count
    while point_mask.sum() < min_points and current_radius < max_radius:
        current_radius += 2.0  # Increase by 2Å increments
        point_mask = min_dists <= current_radius

    return point_mask


def get_largest_component_pointcloud(
    vertex_mask: np.ndarray, verts: np.ndarray, eps: float = 3.0, min_points: int = 10
) -> np.ndarray:
    """
    Extract largest connected component from point cloud using DBSCAN clustering.
    This removes small artifacts/disconnected pieces.

    Args:
        vertex_mask: boolean mask of selected vertices
        verts: all vertices (N, 3)
        eps: DBSCAN eps parameter (distance threshold for clustering)
        min_points: DBSCAN min_points parameter

    Returns: refined boolean mask
    """
    if not HAS_OPEN3D:
        # Without Open3D, just return the original mask
        return vertex_mask

    masked_indices = np.where(vertex_mask)[0]
    if len(masked_indices) == 0:
        return vertex_mask

    # Subset vertices for masked region
    masked_verts = verts[masked_indices]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(masked_verts)

    # Cluster using DBSCAN
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False)
    )

    if len(labels) == 0 or np.all(labels == -1):
        # No clusters found or all points are noise, return original mask
        return vertex_mask

    # Find largest cluster (excluding noise label -1)
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return vertex_mask

    largest_cluster_label = unique_labels[np.argmax(counts)]

    # Get points belonging to largest cluster
    largest_cluster_mask = labels == largest_cluster_label

    # Map back to original vertex indices
    new_mask = np.zeros(len(verts), dtype=bool)
    new_mask[masked_indices[largest_cluster_mask]] = True

    return new_mask


def subset_surface(surface: SurfaceObject, vertex_mask: np.ndarray) -> SurfaceObject:
    """
    Subset dmasif surface (point cloud) based on vertex mask.

    Note: dmasif surfaces don't need spectral operators (no mass, L, evals, evecs, etc.)

    Returns: new SurfaceObject with subsetted data
    """
    vertex_mask_torch = torch.from_numpy(vertex_mask)

    # Subset vertices
    new_verts = surface.verts[vertex_mask_torch] if surface.verts is not None else None

    # Subset vertex normals
    new_vnormals = (
        surface.vnormals[vertex_mask_torch] if surface.vnormals is not None else None
    )

    # Subset batch vector
    new_batch = surface.batch[vertex_mask_torch] if surface.batch is not None else None

    # Create new SurfaceObject (point cloud, no faces)
    new_surface = SurfaceObject(
        verts=new_verts,
        faces=None,  # Point cloud, no mesh connectivity
        vnormals=new_vnormals,
        batch=new_batch,
    )

    # Subset features if they exist
    if hasattr(surface, "features") and surface.features is not None:
        from alphasurf.protein.features import Features

        new_num_nodes = len(new_verts)
        old_features = surface.features
        old_num_nodes = len(surface.verts)

        new_features = Features(num_nodes=new_num_nodes)

        if (
            hasattr(old_features, "flat_features")
            and old_features.flat_features is not None
        ):
            new_features.flat_features = old_features.flat_features[vertex_mask_torch]

        if (
            hasattr(old_features, "named_features")
            and old_features.named_features is not None
        ):
            new_features.named_features = {}
            for key, value in old_features.named_features.items():
                if isinstance(value, (torch.Tensor, np.ndarray)) and hasattr(
                    value, "shape"
                ):
                    if value.shape[0] == old_num_nodes:
                        new_features.named_features[key] = value[vertex_mask_torch]
                    else:
                        new_features.named_features[key] = value
                else:
                    new_features.named_features[key] = value

        if (
            hasattr(old_features, "named_one_hot_features")
            and old_features.named_one_hot_features is not None
        ):
            new_features.named_one_hot_features = {}
            new_features.named_one_hot_features_nclasses = {}
            for key, value in old_features.named_one_hot_features.items():
                if isinstance(value, (torch.Tensor, np.ndarray)) and hasattr(
                    value, "shape"
                ):
                    if value.shape[0] == old_num_nodes:
                        new_features.named_one_hot_features[key] = value[
                            vertex_mask_torch
                        ]
                    else:
                        new_features.named_one_hot_features[key] = value
                else:
                    new_features.named_one_hot_features[key] = value
                new_features.named_one_hot_features_nclasses[key] = (
                    old_features.named_one_hot_features_nclasses[key]
                )

        if hasattr(old_features, "res_map") and old_features.res_map is not None:
            new_features.res_map = old_features.res_map[vertex_mask_torch]
            new_features.num_res = int(new_features.res_map.max()) + 1

        new_features.possible_nums = {new_num_nodes}
        if hasattr(new_features, "num_res"):
            new_features.possible_nums.add(new_features.num_res)

        new_surface.features = new_features

    return new_surface


def process_single_patch(args: Tuple) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single patch: subset dmasif surface.

    Returns: (patch_name, success, error_message)
    """
    patch_path, surface_dir, out_surf_dir, radius, min_points, use_clustering, eps = (
        args
    )

    try:
        # Load patch
        patch = load_patch(patch_path)
        patch_name = patch["patch_name"]

        # Derive protein name from patch (e.g., "6HEG_A_patch_0_ADP" -> "6HEG_A")
        if "_patch_" in patch_name:
            protein_name = patch_name.split("_patch_")[0]
        else:
            parts = patch_name.split("_")
            if len(parts) >= 2:
                protein_name = "_".join(parts[:2])
            else:
                protein_name = parts[0]

        # Find corresponding surface file
        surface_path = os.path.join(surface_dir, f"{protein_name}.pt")

        if not os.path.exists(surface_path):
            return patch_name, False, f"Surface not found: {surface_path}"

        # Load full surface
        surface = load_surface(surface_path)

        # Get surface vertices from SurfaceObject
        if surface.verts is None:
            return patch_name, False, "Surface has no vertices"

        surface_verts = surface.verts.cpu().numpy()

        # Step 1: Find points within radius of patch (adaptively increases radius if needed)
        point_mask = find_points_near_patch(
            surface_verts, patch["verts"], radius, min_points
        )

        if point_mask.sum() == 0:
            return patch_name, False, "No points found within radius"

        # Step 2: Optionally extract largest connected component using point cloud clustering
        if use_clustering and HAS_OPEN3D:
            point_mask = get_largest_component_pointcloud(
                point_mask, surface_verts, eps=eps
            )

            if point_mask.sum() == 0:
                return patch_name, False, "No points after clustering"

            # If clustering removed too many points, try expanding radius
            current_radius = radius
            max_radius = 20.0
            while point_mask.sum() < min_points and current_radius < max_radius:
                current_radius += 2.0
                point_mask = find_points_near_patch(
                    surface_verts, patch["verts"], current_radius, min_points=0
                )
                point_mask = get_largest_component_pointcloud(
                    point_mask, surface_verts, eps=eps
                )
                if point_mask.sum() == 0:
                    break

        if point_mask.sum() < min_points:
            return (
                patch_name,
                False,
                f"Insufficient points: {point_mask.sum()} < {min_points}",
            )

        # Step 3: Subset surface
        subsetted_surface = subset_surface(surface, point_mask)

        # Step 4: Save
        out_surf_path = os.path.join(out_surf_dir, f"{patch_name}.pt")
        os.makedirs(out_surf_dir, exist_ok=True)
        torch.save(subsetted_surface, out_surf_path)

        return patch_name, True, None

    except Exception as e:
        import traceback

        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (
            Path(patch_path).stem if "patch_path" in locals() else "unknown",
            False,
            error_msg,
        )


class PatchDmasifPreprocessDataset:
    """Dataset wrapper for parallel processing."""

    def __init__(
        self,
        patch_dir: str,
        surface_dir: str,
        out_surf_dir: str,
        radius: float = 6.0,
        min_points: int = 50,
        use_clustering: bool = True,
        eps: float = 3.0,
        log_file: str = None,
    ):
        self.patch_dir = patch_dir
        self.surface_dir = surface_dir
        self.out_surf_dir = out_surf_dir
        self.radius = radius
        self.min_points = min_points
        self.use_clustering = use_clustering
        self.eps = eps
        self.log_file = log_file

        # Find all patch files
        self.patch_files = [
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir)
            if f.endswith(".npz")
        ]

        os.makedirs(out_surf_dir, exist_ok=True)

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        args = (
            self.patch_files[idx],
            self.surface_dir,
            self.out_surf_dir,
            self.radius,
            self.min_points,
            self.use_clustering,
            self.eps,
        )
        patch_name, success, error = process_single_patch(args)

        if not success and self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{patch_name}: {error}\n")

        return success


def main():
    """Main preprocessing function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess patches to create subsetted dmasif surfaces"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Data directory containing surfaces_dmasif/ subdirectory",
    )
    parser.add_argument(
        "--patch_dir",
        type=str,
        default=None,
        help="Directory containing .npz patch files (default: data_dir/../dataset_MasifLigand)",
    )
    parser.add_argument(
        "--surface_dir",
        type=str,
        default=None,
        help="Directory containing full dmasif surfaces (default: data_dir/surfaces_dmasif)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for patch surfaces (default: data_dir/surfaces_patches_dmasif)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Initial radius in Angstroms for point selection (default: 6.0, adaptive)",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=50,
        help="Minimum points needed per patch (default: 50)",
    )
    parser.add_argument(
        "--use_clustering",
        action="store_true",
        default=True,
        help="Use DBSCAN clustering to extract largest component (default: True)",
    )
    parser.add_argument(
        "--no_clustering",
        dest="use_clustering",
        action="store_false",
        help="Disable clustering (just use distance-based selection)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=3.0,
        help="DBSCAN eps parameter for clustering (default: 3.0)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(8, cpu_count))",
    )

    args = parser.parse_args()

    # Configuration
    data_dir = os.path.normpath(args.data_dir)

    # Default directories
    if args.patch_dir is None:
        # Try to find patch directory relative to data_dir
        base_dir = os.path.dirname(data_dir)
        patch_dir = os.path.join(base_dir, "dataset_MasifLigand")
    else:
        patch_dir = os.path.normpath(args.patch_dir)

    if args.surface_dir is None:
        surface_dir = os.path.join(data_dir, "surfaces_dmasif")
    else:
        surface_dir = os.path.normpath(args.surface_dir)

    if args.out_dir is None:
        out_surf_dir = os.path.join(data_dir, "surfaces_patches_dmasif")
    else:
        out_surf_dir = os.path.normpath(args.out_dir)

    log_file = os.path.join(data_dir, "preprocess_patches_dmasif_errors.log")

    # Parameters
    num_workers = (
        args.num_workers if args.num_workers else min(8, multiprocessing.cpu_count())
    )

    # Clear log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)

    print("Dmasif patch preprocessing configuration:")
    print(f"  Data dir:      {data_dir}")
    print(f"  Patch dir:     {patch_dir}")
    print(f"  Surface dir:   {surface_dir}")
    print(f"  Output dir:    {out_surf_dir}")
    print(f"  Error log:     {log_file}")
    print(f"  Initial radius: {args.radius} Å (adaptive)")
    print(f"  Min points:    {args.min_points}")
    print(f"  Use clustering: {args.use_clustering}")
    if args.use_clustering:
        print(f"  DBSCAN eps:    {args.eps}")
    print(f"  Workers:       {num_workers}")
    print()

    # Check input directories exist
    if not os.path.exists(patch_dir):
        print(f"✘ Error: Patch directory not found: {patch_dir}")
        print("  Please specify --patch_dir or ensure dataset_MasifLigand exists")
        return

    if not os.path.exists(surface_dir):
        print(f"✘ Error: Surface directory not found: {surface_dir}")
        print("  Please run preprocess_dmasif.py first to generate surfaces")
        return

    # Create dataset
    dataset = PatchDmasifPreprocessDataset(
        patch_dir=patch_dir,
        surface_dir=surface_dir,
        out_surf_dir=out_surf_dir,
        radius=args.radius,
        min_points=args.min_points,
        use_clustering=args.use_clustering,
        eps=args.eps,
        log_file=log_file,
    )

    print(f"Found {len(dataset)} patches to process\n")

    if len(dataset) == 0:
        print("✘ No patches found to process")
        return

    # Process with multiprocessing
    do_all_simple(dataset, num_workers=num_workers)

    # Count errors from log file
    num_errors = 0
    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        with open(log_file, "r") as f:
            num_errors = len(f.readlines())

    print("\n✔ Preprocessing complete!")
    print(f"  Surfaces saved to: {out_surf_dir}")
    print(f"  Successful: {len(dataset) - num_errors}/{len(dataset)}")

    if num_errors > 0:
        print(f"  Failed: {num_errors}")
        print(f"  Errors logged to: {log_file}")
    else:
        print("  No errors encountered")


if __name__ == "__main__":
    main()
