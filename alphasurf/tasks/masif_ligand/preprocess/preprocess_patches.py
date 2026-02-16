"""
Preprocessing script to create subsetted surfaces from patches.

For each .npz patch file:
1. Load patch vertices (defines the region of interest)
2. Load full surface mesh
3. Find all surface vertices within 6Å of any patch vertex
4. Use Open3D to extract largest connected component
5. Save subsetted surface
"""

import os
import sys
import warnings
import numpy as np
import torch
import multiprocessing
from typing import Dict, Tuple, Optional
import open3d as o3d

# Suppress warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, "..", "..", ".."))

from alphasurf.utils.python_utils import do_all_simple
from alphasurf.protein.surfaces import SurfaceObject
from alphasurf.protein.create_operators import compute_operators


def load_patch(patch_path: str) -> Dict:
    """Load patch vertices and faces from npz file."""
    data = np.load(patch_path, allow_pickle=True)
    return {
        "verts": data["pkt_verts"],
        "faces": data["pkt_faces"].astype(int),
        "patch_name": os.path.basename(patch_path).replace(".npz", ""),
    }


def load_surface(surface_path: str) -> SurfaceObject:
    """Load full surface from .pt file."""
    return torch.load(surface_path, weights_only=False)


def compute_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between two sets of points.
    Returns array of shape (len(points_a),) with minimum distance to points_b.
    """
    # For efficiency with large point sets
    min_dists = np.full(len(points_a), np.inf)

    for i, pt in enumerate(points_a):
        dists = np.linalg.norm(points_b - pt, axis=1)
        min_dists[i] = np.min(dists)

    return min_dists


def find_vertices_near_patch(
    surface_verts: np.ndarray,
    patch_verts: np.ndarray,
    radius: float = 6.0,
    min_vertices: int = 130,
    max_radius: float = 20.0,
) -> np.ndarray:
    """
    Find all surface vertices within radius Angstroms of any patch vertex.
    Adaptively increases radius to ensure at least min_vertices are selected.

    Returns: boolean mask of shape (n_surface_verts,)
    """
    # Compute minimum distance from each surface vertex to patch (only once)
    min_dists = compute_distances(surface_verts, patch_verts)

    # Start with initial radius
    current_radius = radius
    vertex_mask = min_dists <= current_radius

    # Adaptively increase radius if needed to meet minimum vertex count
    while vertex_mask.sum() < min_vertices and current_radius < max_radius:
        current_radius += 2.0  # Increase by 2Å increments
        vertex_mask = min_dists <= current_radius

    return vertex_mask


def get_largest_component(
    vertex_mask: np.ndarray, verts: np.ndarray, faces: np.ndarray
) -> np.ndarray:
    """
    Given a vertex mask, extract the largest connected component using Open3D.
    This removes small artifacts/disconnected pieces.

    Returns: refined boolean mask
    """
    masked_indices = np.where(vertex_mask)[0]
    if len(masked_indices) == 0:
        return vertex_mask

    # Create mapping from old to new vertex indices for the masked subset
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(masked_indices)}

    # Subset vertices and faces for masked region
    masked_verts = verts[masked_indices]

    # Filter faces to only include those with all vertices in the mask
    masked_faces = []
    for face in faces:
        if all(v in old_to_new for v in face):
            new_face = [old_to_new[v] for v in face]
            masked_faces.append(new_face)

    if len(masked_faces) == 0:
        # No faces, just return the mask as-is
        return vertex_mask

    masked_faces = np.array(masked_faces, dtype=np.int32)

    # Create Open3D mesh from masked vertices and faces
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(masked_verts)
    mesh.triangles = o3d.utility.Vector3iVector(masked_faces)

    # Cluster connected components
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles()
    )
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if len(cluster_n_triangles) == 0:
        # No clusters, return original mask
        return vertex_mask

    # Find largest cluster
    largest_cluster_idx = np.argmax(cluster_n_triangles)

    # Get faces belonging to largest cluster
    largest_cluster_faces_mask = triangle_clusters == largest_cluster_idx
    largest_cluster_face_indices = np.where(largest_cluster_faces_mask)[0]

    # Get all vertices used in largest cluster faces
    vertices_in_largest = set()
    for face_idx in largest_cluster_face_indices:
        for v_idx in masked_faces[face_idx]:
            vertices_in_largest.add(v_idx)

    # Map back to original vertex indices
    vertices_in_largest_original = {masked_indices[v] for v in vertices_in_largest}

    # Create new mask
    new_mask = np.zeros(len(verts), dtype=bool)
    new_mask[list(vertices_in_largest_original)] = True

    return new_mask


def subset_surface(surface: SurfaceObject, vertex_mask: np.ndarray) -> SurfaceObject:
    """
    Subset surface based on vertex mask.

    Returns: new SurfaceObject with subsetted data
    """
    vertex_indices = np.where(vertex_mask)[0]
    vertex_mask_torch = torch.from_numpy(vertex_mask)

    # Create mapping from old to new vertex indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}

    # Subset vertices
    new_verts = surface.verts[vertex_mask_torch] if surface.verts is not None else None

    # Subset faces and remap indices
    if surface.faces is not None:
        faces_np = (
            surface.faces.cpu().numpy()
            if isinstance(surface.faces, torch.Tensor)
            else surface.faces
        )
        valid_faces = []
        for face in faces_np:
            if all(v in old_to_new for v in face):
                new_face = [old_to_new[v] for v in face]
                valid_faces.append(new_face)

        if valid_faces:
            new_faces = torch.tensor(valid_faces, dtype=surface.faces.dtype)
        else:
            new_faces = torch.empty((0, 3), dtype=surface.faces.dtype)
    else:
        new_faces = None

    # Subset vertex normals
    new_vnormals = (
        surface.vnormals[vertex_mask_torch] if surface.vnormals is not None else None
    )

    # Recompute spectral operators for the subsetted surface
    # NOTE: We compute operators on the subset with k_eig=128 for consistency.
    # This gives accurate operators for the actual patch geometry (with proper boundary conditions).
    # Small patches (< 130 vertices) will fail eigendecomposition and be logged as errors.
    verts_np = (
        new_verts.cpu().numpy() if isinstance(new_verts, torch.Tensor) else new_verts
    )
    faces_np = (
        new_faces.cpu().numpy() if isinstance(new_faces, torch.Tensor) else new_faces
    )
    vnormals_np = (
        new_vnormals.cpu().numpy()
        if isinstance(new_vnormals, torch.Tensor)
        else new_vnormals
    )

    frames, new_mass, new_L, new_evals, new_evecs, new_gradX, new_gradY = (
        compute_operators(verts_np, faces_np, normals=vnormals_np, k_eig=128)
    )

    # Ensure evals and evecs are tensors (not numpy arrays or lists) for proper batching
    # Keep sparse matrices (mass, L, gradX, gradY) as scipy sparse - they'll be converted later via from_numpy()
    def to_tensor(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        if isinstance(x, list):
            return torch.tensor(x, dtype=torch.float32)
        return torch.tensor(x, dtype=torch.float32)

    # Only convert dense arrays to tensors; keep sparse matrices as scipy sparse
    new_evals = to_tensor(new_evals)
    new_evecs = to_tensor(new_evecs)
    # Keep mass, L, gradX, gradY as scipy sparse matrices (they can be saved with torch.save)

    # Create new SurfaceObject
    new_surface = SurfaceObject(
        verts=new_verts,
        faces=new_faces,
        vnormals=new_vnormals,
        mass=new_mass,
        L=new_L,
        evals=new_evals,
        evecs=new_evecs,
        gradX=new_gradX,
        gradY=new_gradY,
    )

    # Subset features from the original surface
    if hasattr(surface, "features") and surface.features is not None:
        from alphasurf.protein.features import Features
        import copy

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

        if (
            hasattr(old_features, "misc_features")
            and old_features.misc_features is not None
        ):
            new_features.misc_features = copy.deepcopy(old_features.misc_features)

        new_surface.features = new_features

    return new_surface


def process_single_patch(args: Tuple) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single patch: subset surface.

    Returns: (patch_name, success, error_message)
    """
    patch_path, surface_dir, out_surf_dir, radius, min_vertices = args

    try:
        # Load patch
        patch = load_patch(patch_path)
        patch_name = patch["patch_name"]

        # Derive protein name from patch (e.g., "6HEG_A_patch_0_ADP" -> "6HEG_A")
        # Format is: {PROTEIN}_patch_{NUM}_{LIGAND}
        if "_patch_" in patch_name:
            protein_name = patch_name.split("_patch_")[0]
        else:
            # Fallback to old logic
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

        # Get faces
        if surface.faces is None:
            return patch_name, False, "Surface has no faces"

        faces = surface.faces.cpu().numpy()

        # Step 1: Find vertices within radius of patch (adaptively increases radius if needed)
        vertex_mask = find_vertices_near_patch(
            surface_verts, patch["verts"], radius, min_vertices
        )

        if vertex_mask.sum() == 0:
            return patch_name, False, "No vertices found within radius"

        # Step 2: Extract largest connected component using Open3D (removes artifacts)
        vertex_mask = get_largest_component(vertex_mask, surface_verts, faces)

        if vertex_mask.sum() == 0:
            return patch_name, False, "No vertices after connected component extraction"

        # Step 3: Check if we still have enough vertices after connected component extraction
        # If not, expand radius further
        current_radius = radius
        max_radius = 20.0
        while vertex_mask.sum() < min_vertices and current_radius < max_radius:
            current_radius += 2.0
            vertex_mask = find_vertices_near_patch(
                surface_verts, patch["verts"], current_radius, min_vertices=0
            )
            vertex_mask = get_largest_component(vertex_mask, surface_verts, faces)
            if vertex_mask.sum() == 0:
                break

        if vertex_mask.sum() < min_vertices:
            return (
                patch_name,
                False,
                f"Insufficient vertices after connected component: {vertex_mask.sum()} < {min_vertices}",
            )

        # Step 4: Subset surface
        subsetted_surface = subset_surface(surface, vertex_mask)

        # Step 5: Save
        out_surf_path = os.path.join(out_surf_dir, f"{patch_name}.pt")
        torch.save(subsetted_surface, out_surf_path)

        return patch_name, True, None

    except Exception as e:
        return patch_path, False, str(e)


class PatchPreprocessDataset:
    """Dataset wrapper for parallel processing."""

    def __init__(
        self,
        patch_dir: str,
        surface_dir: str,
        out_surf_dir: str,
        radius: float = 6.0,
        min_vertices: int = 130,
        log_file: str = None,
    ):
        self.patch_dir = patch_dir
        self.surface_dir = surface_dir
        self.out_surf_dir = out_surf_dir
        self.radius = radius
        self.min_vertices = min_vertices
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
            self.min_vertices,
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
        description="Preprocess patches to create subsetted surfaces"
    )
    parser.add_argument(
        "--data_subdir",
        type=str,
        required=True,
        help="Data subdirectory (e.g., msms_01, alpha0, alpha1, alpha10, alpha25)",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=6.0,
        help="Initial radius in Angstroms for vertex selection (default: 6.0, adaptive)",
    )
    parser.add_argument(
        "--min_vertices",
        type=int,
        default=130,
        help="Minimum vertices needed for eigendecomposition (default: 130)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: min(8, cpu_count))",
    )

    args = parser.parse_args()

    # Configuration
    script_dir = os.path.dirname(os.path.realpath(__file__))
    base_data_dir = os.path.normpath(
        os.path.join(script_dir, "..", "..", "..", "data", "masif_ligand")
    )
    data_subdir_path = os.path.join(base_data_dir, args.data_subdir)

    if not os.path.exists(data_subdir_path):
        print(f"✘ Error: Data subdirectory not found: {data_subdir_path}")
        return

    # Input directories
    patch_dir = os.path.join(base_data_dir, "dataset_MasifLigand")

    # Find surface directory (try different naming patterns)
    possible_surface_dirs = [
        os.path.join(data_subdir_path, "surfaces_full_msms_0.1_False"),
        os.path.join(data_subdir_path, "surfaces_full_alpha_complex_1.0_False"),
    ]

    surface_dir = None
    for surf_dir in possible_surface_dirs:
        if os.path.exists(surf_dir):
            surface_dir = surf_dir
            break

    if surface_dir is None:
        print(f"✘ Error: No surface directory found in {data_subdir_path}")
        print(f"  Looked for: {possible_surface_dirs}")
        return

    # Output directories
    out_surf_dir = os.path.join(data_subdir_path, "surfaces_patches")
    log_file = os.path.join(data_subdir_path, "preprocess_patches_errors.log")

    # Parameters
    radius = args.radius
    min_vertices = args.min_vertices
    num_workers = (
        args.num_workers if args.num_workers else min(8, multiprocessing.cpu_count())
    )

    # Clear log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)

    print("Patch preprocessing configuration:")
    print(f"  Data subdir:  {args.data_subdir}")
    print(f"  Patch dir:    {patch_dir}")
    print(f"  Surface dir:  {surface_dir}")
    print(f"  Output surf:  {out_surf_dir}")
    print(f"  Error log:    {log_file}")
    print(f"  Initial radius: {radius} Å (adaptive)")
    print(f"  Min vertices: {min_vertices}")
    print(f"  Workers:      {num_workers}")
    print()

    # Check input directories exist
    if not os.path.exists(patch_dir):
        print(f"✘ Error: Patch directory not found: {patch_dir}")
        print("  Please adjust patch_dir in the script to point to your .npz files")
        return

    if not os.path.exists(surface_dir):
        print(f"✘ Error: Surface directory not found: {surface_dir}")
        return

    # Create dataset
    dataset = PatchPreprocessDataset(
        patch_dir=patch_dir,
        surface_dir=surface_dir,
        out_surf_dir=out_surf_dir,
        radius=radius,
        min_vertices=min_vertices,
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
