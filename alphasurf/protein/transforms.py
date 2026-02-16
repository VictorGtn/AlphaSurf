"""
Modular transforms for protein data.

These transforms are applied DURING surface/graph generation so that
computed features (operators, edges, etc.) reflect the transformed geometry.

Transform Order (critical):
1. Parse PDB -> raw arrays
2. NoiseAugmentor.noise_arrays() -> noised atom positions (for joint/independent modes)
3. Generate mesh from (noised) atoms
4. PatchExtractor.extract_patch() -> subset mesh to binding site
5. add_mesh_noise() -> displace vertices along normals (independent mode only)
6. Compute operators (mass, L, evals, evecs) on final mesh
7. Compute features on final geometry
"""

import os
from typing import Optional, Tuple

import numpy as np


class NoiseAugmentor:
    """
    Handles noise augmentation for protein coordinates.

    Supports two modes:
    - 'independent': Noise graph coordinates (sigma_graph) and mesh vertices (sigma_mesh)
                     independently. Graph noise applied to atom coords before graph build.
                     Mesh noise applied to vertices after patch extraction.
    - 'joint': Noise atom coordinates once, then use for BOTH mesh and graph generation.
               This propagates coordinate noise naturally to both representations.
    - 'none': No noise augmentation (default).

    All operations create new arrays (no in-place modification).
    """

    def __init__(
        self,
        mode: str = "none",
        sigma_graph: float = 0.3,
        sigma_mesh: float = 0.3,
        clip_sigma: float = None,
        alpha_min: float = 0.0,
        alpha_max: float = 5.0,
    ):
        """
        Args:
            mode: 'none', 'independent', 'joint', or 'alpha'
            sigma_graph: Noise sigma for atom/graph coordinates
            sigma_mesh: Noise sigma for mesh vertices (only used in 'independent' mode)
            clip_sigma: Clip noise to ±clip_sigma*sigma (None to disable)
            alpha_min: Minimum alpha value for alpha noise (default: 0.0)
            alpha_max: Maximum alpha value for alpha noise (default: 20.0)
        """
        self.mode = mode.lower()
        self.sigma_graph = sigma_graph
        self.sigma_mesh = sigma_mesh
        self.clip_sigma = clip_sigma
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        if self.mode not in ("none", "independent", "joint", "alpha"):
            raise ValueError(
                f"Unknown noise mode: {mode}. Must be 'none', 'independent', 'joint', or 'alpha'."
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def sample_alpha_value(self) -> float:
        """
        Sample a random alpha value uniformly from [alpha_min, alpha_max].

        Returns:
            Random alpha value for alpha complex generation
        """
        if self.mode != "alpha":
            return None
        return np.random.uniform(self.alpha_min, self.alpha_max)

    def noise_arrays(self, parsed_arrays: tuple) -> tuple:
        """
        Apply Gaussian noise to atom positions in parsed PDB arrays.

        Creates a NEW tuple with a NEW atom_pos array. Original is never modified.

        Args:
            parsed_arrays: Tuple from parse_pdb_path containing:
                          (amino_types, atom_chain_id, atom_amino_id, atom_names,
                           atom_types, atom_pos, atom_charge, atom_radius, ...)

        Returns:
            New tuple with noised atom_pos at index 5
        """
        parsed_list = list(parsed_arrays)
        atom_pos = parsed_list[5].copy()  # Explicit copy

        noise = np.random.randn(*atom_pos.shape) * self.sigma_graph

        if self.clip_sigma is not None:
            np.clip(
                noise,
                -self.clip_sigma * self.sigma_graph,
                self.clip_sigma * self.sigma_graph,
                out=noise,
            )

        atom_pos = (atom_pos + noise).astype(np.float32)
        parsed_list[5] = atom_pos

        return tuple(parsed_list)

    def prepare_arrays(
        self, parsed_arrays: tuple
    ) -> Tuple[tuple, tuple, Optional[float], bool]:
        """
        Prepare arrays for surface and graph generation based on noise mode.

        Encapsulates the logic for joint vs independent noise application.

        Args:
            parsed_arrays: The original parsed PDB arrays (clean)

        Returns:
            Tuple of:
            - parsed_for_surface: Arrays to use for surface generation
            - parsed_for_graph: Arrays to use for graph generation
            - alpha_override: Optional alpha value (for 'alpha' mode)
            - apply_mesh_noise: Whether to apply mesh noise later (for 'independent' mode)
        """
        if self.mode == "joint":
            noised = self.noise_arrays(parsed_arrays)
            return noised, noised, None, False
        elif self.mode == "independent":
            # Graph gets atom noise, surface gets clean atoms (will apply mesh noise later)
            noised_graph = self.noise_arrays(parsed_arrays)
            return parsed_arrays, noised_graph, None, True
        elif self.mode == "alpha":
            return parsed_arrays, parsed_arrays, self.sample_alpha_value(), False
        else:
            # None or unknown
            return parsed_arrays, parsed_arrays, None, False


class PatchExtractor:
    """
    Extracts patch regions from full surfaces using reference patch data.

    Uses pkt_verts from dataset_MasifLigand/*.npz files to identify
    binding site regions, then subsets the generated mesh to only those
    vertices within a radius of the reference patch.

    This is called BEFORE computing operators (mass, L, evals, evecs),
    so operators are computed only on the smaller patch (~1500 verts)
    instead of the full surface (~10000+ verts).
    """

    def __init__(
        self,
        patch_dir: str,
        radius: float = 6.0,
        min_verts: int = 140,
        max_radius: float = 12.0,
    ):
        """
        Args:
            patch_dir: Path to directory containing *.npz patch files
            radius: Initial distance threshold in Angstroms for including vertices
            min_verts: Minimum number of vertices required in patch
            max_radius: Maximum distance threshold when iterating
        """
        self.patch_dir = patch_dir
        self.radius = radius
        self.min_verts = min_verts
        self.max_radius = max_radius

    def get_patch_vertices(self, pocket_name: str) -> Optional[np.ndarray]:
        """Load reference patch vertices from NPZ file."""
        npz_path = os.path.join(self.patch_dir, f"{pocket_name}.npz")
        if not os.path.exists(npz_path):
            return None

        try:
            data = np.load(npz_path, allow_pickle=True)
            return data["pkt_verts"]
        except Exception as e:
            print(f"Failed to load patch {pocket_name}: {e}")
            return None

    def extract_patch(
        self, verts: np.ndarray, faces: np.ndarray, pocket_name: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Subset a mesh to just the patch region around the binding site.

        Args:
            verts: (N, 3) array of surface vertices
            faces: (M, 3) array of face indices
            pocket_name: Name like "1ABC_A_patch_0_HEM"

        Returns:
            Tuple of (patch_verts, patch_faces) with remapped indices,
            or None if extraction fails
        """
        ref_verts = self.get_patch_vertices(pocket_name)
        if ref_verts is None:
            return None

        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return self._extract_patch_brute_force(verts, faces, ref_verts)

        tree = cKDTree(ref_verts)
        distances, _ = tree.query(verts, k=1)

        current_radius = self.radius
        while current_radius <= self.max_radius:
            vertex_mask = distances <= current_radius

            if vertex_mask.sum() > 0:
                vertex_mask_cc = self._get_largest_component(vertex_mask, verts, faces)

                if vertex_mask_cc.sum() >= self.min_verts:
                    result = self._subset_mesh(verts, faces, vertex_mask_cc)
                    if result is not None:
                        return result

            current_radius += 2.0

        return None

    def _get_largest_component(
        self, vertex_mask: np.ndarray, verts: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """Extract largest connected component from masked mesh region."""
        try:
            import open3d as o3d
        except ImportError:
            return vertex_mask

        masked_indices = np.where(vertex_mask)[0]
        if len(masked_indices) == 0:
            return vertex_mask

        old_to_new = {
            old_idx: new_idx for new_idx, old_idx in enumerate(masked_indices)
        }
        masked_verts = verts[masked_indices]

        masked_faces = []
        for face in faces:
            if all(int(v) in old_to_new for v in face):
                new_face = [old_to_new[int(v)] for v in face]
                masked_faces.append(new_face)

        if len(masked_faces) == 0:
            return vertex_mask

        masked_faces = np.array(masked_faces, dtype=np.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(masked_verts)
        mesh.triangles = o3d.utility.Vector3iVector(masked_faces)

        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) == 0:
            return vertex_mask

        largest_cluster_idx = np.argmax(cluster_n_triangles)
        largest_cluster_faces_mask = triangle_clusters == largest_cluster_idx
        largest_cluster_face_indices = np.where(largest_cluster_faces_mask)[0]

        vertices_in_largest = set()
        for face_idx in largest_cluster_face_indices:
            for v_idx in masked_faces[face_idx]:
                vertices_in_largest.add(v_idx)

        vertices_in_largest_original = {masked_indices[v] for v in vertices_in_largest}

        new_mask = np.zeros(len(verts), dtype=bool)
        new_mask[list(vertices_in_largest_original)] = True

        return new_mask

    def _extract_patch_brute_force(
        self, verts: np.ndarray, faces: np.ndarray, ref_verts: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Fallback extraction without scipy."""
        min_dists = np.full(len(verts), np.inf)
        for i, v in enumerate(verts):
            dists = np.linalg.norm(ref_verts - v, axis=1)
            min_dists[i] = dists.min()

        vertex_mask = min_dists <= self.radius
        if vertex_mask.sum() == 0:
            return None

        return self._subset_mesh(verts, faces, vertex_mask)

    def _subset_mesh(
        self, verts: np.ndarray, faces: np.ndarray, vertex_mask: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Create subset mesh with only masked vertices and valid faces."""
        kept_indices = np.where(vertex_mask)[0]
        old_to_new = np.full(len(verts), -1, dtype=np.int64)
        old_to_new[kept_indices] = np.arange(len(kept_indices))

        faces_int = faces.astype(np.int64)
        valid_face_mask = np.all(vertex_mask[faces_int], axis=1)
        kept_faces = faces_int[valid_face_mask]

        if len(kept_faces) == 0:
            return None

        remapped_faces = old_to_new[kept_faces]
        return verts[kept_indices].astype(np.float32), remapped_faces.astype(np.int64)


def add_mesh_noise(
    verts: np.ndarray,
    faces: np.ndarray,
    sigma: float = 0.3,
    clip_sigma: Optional[float] = 3.0,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Add noise to mesh vertices along their normals.

    Args:
        verts: (N, 3) array of vertex positions
        faces: (M, 3) array of face indices
        sigma: Standard deviation of displacement in Angstroms
        clip_sigma: Clip noise to ±clip_sigma*sigma (None to disable)
        normals: (N, 3) pre-computed normals (auto-computed if None)

    Returns:
        (N, 3) array of displaced vertex positions (new array)
    """
    import igl

    if normals is None:
        normals = igl.per_vertex_normals(verts, faces)

    noise = np.random.randn(len(verts), 1) * sigma

    if clip_sigma is not None:
        noise = np.clip(noise, -clip_sigma * sigma, clip_sigma * sigma)

    return (verts + normals * noise).astype(np.float32)
