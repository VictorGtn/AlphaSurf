"""
On-the-fly data loading for MaSIF-Ligand.

This module provides data loaders that generate surfaces, graphs, and ESM embeddings
on-the-fly during training, eliminating the need for a separate preprocessing step.

Supports optional noise augmentation with two modes:
- 'independent': Noise graph coordinates and mesh vertices independently (two sigmas)
- 'joint': Noise atom coordinates, regenerate mesh from noised positions
"""

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, "..", "..", ".."))

from alphasurf.protein.create_surface import add_surface_noise
from alphasurf.protein.graphs import parse_pdb_path
from alphasurf.protein.residue_graph import ResidueGraphBuilder
from alphasurf.protein.surfaces import SurfaceObject
from alphasurf.utils.data_utils import AtomBatch, update_model_input_dim
from data_loader import get_systems_from_ligands


def add_graph_noise(
    node_pos: torch.Tensor, sigma: float, clip_sigma: float = 3.0
) -> torch.Tensor:
    """
    Add isotropic Gaussian noise to graph node positions.

    All operations on CPU as requested.

    Args:
        node_pos: (N, 3) tensor of node positions
        sigma: Standard deviation of displacement in Angstroms
        clip_sigma: Clip noise to ±clip_sigma*sigma to prevent extreme outliers

    Returns:
        node_pos_noisy: (N, 3) tensor of displaced node positions
    """
    noise = torch.randn_like(node_pos) * sigma
    if clip_sigma is not None:
        noise = torch.clamp(noise, -clip_sigma * sigma, clip_sigma * sigma)
    return node_pos + noise


class NoiseAugmentor:
    """
    Handles noise augmentation for on-the-fly training.

    Supports two modes:
    - 'independent': Noise graph coordinates (sigma_graph) and mesh vertices (sigma_mesh)
                     independently. Mesh noise is applied along vertex normals.
    - 'joint': Noise atom coordinates first, then regenerate the mesh from the
               noised atomic positions. This propagates coordinate noise naturally.
    - 'none': No noise augmentation (default).

    All operations are performed on CPU.
    """

    def __init__(
        self,
        noise_mode: str = "none",
        sigma_graph: float = 0.3,
        sigma_mesh: float = 0.3,
        clip_sigma: float = 3.0,
    ):
        """
        Args:
            noise_mode: 'none', 'independent', or 'joint'
            sigma_graph: Noise sigma for graph coordinates (used in both modes)
            sigma_mesh: Noise sigma for mesh vertices (only used in 'independent' mode)
            clip_sigma: Clip noise to ±clip_sigma*sigma
        """
        self.noise_mode = noise_mode.lower()
        self.sigma_graph = sigma_graph
        self.sigma_mesh = sigma_mesh
        self.clip_sigma = clip_sigma

        if self.noise_mode not in ("none", "independent", "joint"):
            raise ValueError(
                f"Unknown noise_mode: {noise_mode}. Must be 'none', 'independent', or 'joint'."
            )

    def apply_independent_noise(
        self,
        surface: SurfaceObject,
        graph: Data,
    ) -> tuple:
        """
        Apply independent noise to graph and mesh.

        - Graph: isotropic noise to node_pos
        - Mesh: noise along vertex normals

        Args:
            surface: SurfaceObject with verts, faces, vnormals
            graph: Graph data object with node_pos

        Returns:
            Tuple of (noised_surface, noised_graph)
        """
        # Noise graph coordinates
        if hasattr(graph, "node_pos") and graph.node_pos is not None:
            graph.node_pos = add_graph_noise(
                graph.node_pos, self.sigma_graph, self.clip_sigma
            )

        # Noise mesh vertices along normals
        if hasattr(surface, "verts") and surface.verts is not None:
            verts_np = (
                surface.verts.cpu().numpy()
                if isinstance(surface.verts, torch.Tensor)
                else surface.verts
            )
            faces_np = (
                surface.faces.cpu().numpy()
                if isinstance(surface.faces, torch.Tensor)
                else surface.faces
            )
            normals_np = (
                surface.vnormals.cpu().numpy()
                if (
                    hasattr(surface, "vnormals")
                    and surface.vnormals is not None
                    and isinstance(surface.vnormals, torch.Tensor)
                )
                else getattr(surface, "vnormals", None)
            )

            verts_noisy = add_surface_noise(
                verts_np,
                faces_np,
                sigma=self.sigma_mesh,
                normals=normals_np,
                clip_sigma=self.clip_sigma,
            )
            surface.verts = torch.from_numpy(verts_noisy).float()
            # Recompute normals after moving vertices
            surface.set_vnormals(force=True)

        return surface, graph

    def noise_graph(self, graph: Data) -> Data:
        """
        Apply isotropic noise to graph node_pos only.

        Used in 'independent' mode when mesh noise is handled separately.

        Args:
            graph: Graph data object with node_pos

        Returns:
            The same graph object with noised node_pos (modified in-place)
        """
        if hasattr(graph, "node_pos") and graph.node_pos is not None:
            graph.node_pos = add_graph_noise(
                graph.node_pos, self.sigma_graph, self.clip_sigma
            )
        return graph

    def noise_parsed_arrays(self, parsed_arrays: tuple) -> tuple:
        """
        Apply noise to parsed atom positions for 'joint' mode.
        WARNING: Modifies atom_pos in-place.
        """
        parsed_list = list(parsed_arrays)
        atom_pos = parsed_list[5]

        noise = np.random.randn(*atom_pos.shape) * self.sigma_graph

        if self.clip_sigma is not None:
            np.clip(
                noise,
                -self.clip_sigma * self.sigma_graph,
                self.clip_sigma * self.sigma_graph,
                out=noise,  # In-place clipping
            )

        atom_pos += noise  # In-place addition
        atom_pos = atom_pos.astype(
            np.float32, copy=False
        )  # Avoid copy if already float32
        parsed_list[5] = atom_pos

        return tuple(parsed_list)


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
            patch_dir: Path to dataset_MasifLigand/ containing *.npz patch files
            radius: Initial distance threshold in Angstroms for including vertices
            min_verts: Minimum number of vertices required in patch. If fewer,
                      radius is increased by 2.0 and extraction retried.
            max_radius: Maximum distance threshold when iterating.
        """
        self.patch_dir = patch_dir
        self.radius = radius
        self.min_verts = min_verts
        self.max_radius = max_radius
        self._cache: Dict[str, np.ndarray] = {}

    def get_patch_vertices(self, pocket_name: str) -> Optional[np.ndarray]:
        """
        Load reference patch vertices from NPZ file.

        Args:
            pocket_name: Name like "1ABC_A_patch_0_HEM"

        Returns:
            (N, 3) array of patch vertex coordinates, or None if not found
        """
        if pocket_name in self._cache:
            return self._cache[pocket_name]

        npz_path = os.path.join(self.patch_dir, f"{pocket_name}.npz")
        if not os.path.exists(npz_path):
            print(f"Patch file not found: {npz_path}")
            return None

        try:
            data = np.load(npz_path, allow_pickle=True)
            patch_verts = data["pkt_verts"]
            self._cache[pocket_name] = patch_verts
            return patch_verts
        except Exception as e:
            print(f"Failed to load patch {pocket_name}: {e}")
            return None

    def extract_patch(
        self, verts: np.ndarray, faces: np.ndarray, pocket_name: str
    ) -> Optional[tuple]:
        """
        Subset a mesh to just the patch region around the binding site.

        Called AFTER mesh generation but BEFORE add_geom_feats() so operators
        are computed on patch only.

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

        # Use scipy KDTree for efficient nearest-neighbor search
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            print("scipy not available, falling back to brute force")
            return self._extract_patch_brute_force(verts, faces, ref_verts)

        tree = cKDTree(ref_verts)
        distances, _ = tree.query(verts, k=1)

        # Iterative extraction: increase radius if patch is too small
        current_radius = self.radius
        while True:
            # Find vertices within radius of any reference patch vertex
            vertex_mask = distances <= current_radius

            if vertex_mask.sum() > 0:
                # Extract largest connected component to remove disconnected fragments
                vertex_mask_cc = self._get_largest_component(vertex_mask, verts, faces)

                # Check if we have enough vertices
                n_verts = vertex_mask_cc.sum()
                if n_verts >= self.min_verts:
                    res = self._subset_mesh(verts, faces, vertex_mask_cc)
                    if res is not None:
                        return res

            # Check if we should continue
            if current_radius >= self.max_radius:
                break

            current_radius += 2.0

        print(
            f"[PATCH] Failed to find valid patch >= {self.min_verts} verts for {pocket_name} (max radius {self.max_radius})"
        )
        return None

    def _get_largest_component(
        self, vertex_mask: np.ndarray, verts: np.ndarray, faces: np.ndarray
    ) -> np.ndarray:
        """
        Extract largest connected component from masked mesh region using Open3D.
        Removes small disconnected fragments.

        Args:
            vertex_mask: (N,) boolean mask of initially selected vertices
            verts: (N, 3) all surface vertices
            faces: (M, 3) all surface face indices

        Returns:
            Refined vertex_mask containing only the largest connected component
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError(
                "Open3D not available, cannot extract largest connected component"
            )

        masked_indices = np.where(vertex_mask)[0]
        if len(masked_indices) == 0:
            return vertex_mask

        # Create mapping from old to new indices for masked region
        old_to_new = {
            old_idx: new_idx for new_idx, old_idx in enumerate(masked_indices)
        }

        # Extract vertices for masked region
        masked_verts = verts[masked_indices]

        # Extract faces where ALL vertices are in the masked region
        masked_faces = []
        for face in faces:
            if all(int(v) in old_to_new for v in face):
                new_face = [old_to_new[int(v)] for v in face]
                masked_faces.append(new_face)

        if len(masked_faces) == 0:
            return vertex_mask

        masked_faces = np.array(masked_faces, dtype=np.int32)

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(masked_verts)
        mesh.triangles = o3d.utility.Vector3iVector(masked_faces)

        # Cluster connected triangles
        triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        if len(cluster_n_triangles) == 0:
            return vertex_mask

        # Find largest cluster
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        largest_cluster_faces_mask = triangle_clusters == largest_cluster_idx
        largest_cluster_face_indices = np.where(largest_cluster_faces_mask)[0]

        # Get vertices in largest cluster
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

    def _extract_patch_brute_force(
        self, verts: np.ndarray, faces: np.ndarray, ref_verts: np.ndarray
    ) -> Optional[tuple]:
        """Fallback extraction without scipy."""
        # Compute min distance from each vertex to reference patch
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
    ) -> tuple:
        """
        Create subset mesh with only masked vertices and valid faces.

        Args:
            verts: (N, 3) original vertices
            faces: (M, 3) original face indices
            vertex_mask: (N,) boolean mask of vertices to keep

        Returns:
            Tuple of (new_verts, new_faces) with remapped indices
        """
        # Get indices of kept vertices
        kept_indices = np.where(vertex_mask)[0]

        # Create mapping from old to new indices
        old_to_new = np.full(len(verts), -1, dtype=np.int64)
        old_to_new[kept_indices] = np.arange(len(kept_indices))

        # Filter faces: keep only faces where ALL 3 vertices are in the subset
        faces_int = faces.astype(np.int64)
        valid_face_mask = np.all(vertex_mask[faces_int], axis=1)
        kept_faces = faces_int[valid_face_mask]

        if len(kept_faces) == 0:
            print("Warning: No valid faces after patch extraction")
            # Return just vertices if no faces
            return verts[kept_indices], np.array([]).reshape(0, 3)

        # Remap face indices to new vertex numbering
        remapped_faces = old_to_new[kept_faces]

        return verts[kept_indices].astype(np.float32), remapped_faces.astype(np.int64)

    def clear_cache(self):
        """Clear the in-memory cache of patch vertices."""
        self._cache.clear()


class OnFlyPDBParser:
    """
    Shared PDB parser that caches parsed arrays to avoid redundant parsing.

    When using alpha_complex method, both surface and graph generation need
    parsed PDB data. This class ensures the PDB is only parsed once.
    """

    def __init__(self, pdb_dir: str, use_cache: bool = True):
        self.pdb_dir = pdb_dir
        self.use_cache = use_cache
        self._cache: Dict[str, tuple] = {}

    def parse(self, protein_name: str) -> Optional[tuple]:
        """
        Parse PDB and return arrays.

        Returns:
            Tuple of parsed arrays, or None on failure
        """
        if self.use_cache and protein_name in self._cache:
            return self._cache[protein_name]

        pdb_path = os.path.join(self.pdb_dir, f"{protein_name}.pdb")
        if not os.path.exists(pdb_path):
            print(f"PDB file not found: {pdb_path}")
            return None

        try:
            try:
                arrays = parse_pdb_path(pdb_path, use_pqr=False)
            except:
                arrays = parse_pdb_path(pdb_path)

            if self.use_cache:
                self._cache[protein_name] = arrays

            return arrays

        except Exception as e:
            print(f"PDB parsing failed for {protein_name}: {e}")
            return None

    def get_pdb_path(self, protein_name: str) -> str:
        """Get the full PDB path for a protein name."""
        return os.path.join(self.pdb_dir, f"{protein_name}.pdb")

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()


class OnFlySurfaceLoader:
    """
    Generates surfaces on-the-fly from PDB files instead of loading from disk.

    Supports optional in-memory caching to avoid regenerating the same surface
    multiple times (especially useful for validation/test sets across epochs).
    """

    def __init__(
        self, config, pdb_dir: str, patch_extractor: Optional[PatchExtractor] = None
    ):
        """
        Args:
            config: Configuration object with surface parameters
            pdb_dir: Path to directory containing PDB files
            patch_extractor: Optional PatchExtractor for extracting binding site patches
        """
        self.config = config
        self.pdb_dir = pdb_dir
        self.patch_extractor = patch_extractor

        # Surface generation parameters
        self.surface_method = getattr(config, "surface_method", "msms")
        self.alpha_value = getattr(config, "alpha_value", 0.1)
        self.face_reduction_rate = getattr(config, "face_reduction_rate", 0.1)
        self.max_vert_number = getattr(config, "max_vert_number", 100000)
        self.use_pymesh = getattr(config, "use_pymesh", False)

        # Feature processing
        self.feat_keys = getattr(config, "feat_keys", "all")
        self.oh_keys = getattr(config, "oh_keys", "all")
        self.use_surfaces = getattr(config, "use_surfaces", True)
        self.use_whole_surfaces = getattr(config, "use_whole_surfaces", True)

        # Caching
        self.use_cache = getattr(config, "use_cache", False)
        self._cache: Dict[str, SurfaceObject] = {}

        # Debug mode: save PLY files for visualization
        self.debug_save_ply = getattr(config, "debug_save_ply", False)
        self.debug_ply_dir = getattr(config, "debug_ply_dir", "/tmp/onfly_debug")
        self._debug_count = 0
        self._debug_max_samples = getattr(config, "debug_max_samples", 5)
        self.debug_exit_after_save = getattr(config, "debug_exit_after_save", False)
        if self.debug_save_ply:
            os.makedirs(self.debug_ply_dir, exist_ok=True)
            print(f"[DEBUG] Saving PLY files to {self.debug_ply_dir}")

    def _save_debug_ply(self, verts: np.ndarray, faces: np.ndarray, name: str):
        """Save mesh as PLY file for debugging/visualization."""
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
        mesh.compute_vertex_normals()

        ply_path = os.path.join(self.debug_ply_dir, f"{name}.ply")
        o3d.io.write_triangle_mesh(ply_path, mesh)
        print(f"[DEBUG] Saved {ply_path} ({len(verts)} verts, {len(faces)} faces)")

    def load(
        self,
        pocket_name: str,
        parsed_arrays: Optional[tuple] = None,
        apply_noise: bool = False,
    ) -> Optional[SurfaceObject]:
        """
        Generate or retrieve a surface for the given pocket/protein.

        Args:
            pocket_name: Name like "1ABC_A_patch_0_HEM"
            parsed_arrays: Optional pre-parsed PDB arrays from OnFlyPDBParser.
                          Used to avoid redundant parsing when using alpha_complex method.

        Returns:
            SurfaceObject with features expanded, or None on failure
        """
        if not self.use_surfaces:
            return Data()

        # Always extract protein name from pocket name for PDB file lookup
        # pocket_name: "1ABC_A_patch_0_HEM" -> protein_name: "1ABC_A"
        if "_patch_" in pocket_name:
            protein_name = pocket_name.split("_patch_")[0]
        else:
            protein_name = pocket_name

        # Check cache
        if self.use_cache and protein_name in self._cache:
            return self._cache[protein_name]

        # Generate surface from PDB
        pdb_path = os.path.join(self.pdb_dir, f"{protein_name}.pdb")
        if not os.path.exists(pdb_path):
            print(f"PDB file not found: {pdb_path}")
            return None

        try:
            # For alpha_complex, use pre-parsed arrays if available
            extra_kwargs = {}
            if self.surface_method == "alpha_complex" and parsed_arrays is not None:
                # parsed_arrays: (amino_types, atom_chain_id, atom_amino_id, atom_names,
                #                 atom_types, atom_pos, atom_charge, atom_radius, ...)
                atom_pos = parsed_arrays[5]
                atom_radius = parsed_arrays[7]
                extra_kwargs["atom_pos"] = atom_pos
                extra_kwargs["atom_radius"] = atom_radius

            # Extract patch BEFORE computing operators (if not using whole surfaces)
            if not self.use_whole_surfaces and self.patch_extractor is not None:
                # Generate mesh only (no operators) - we'll compute on patch
                from alphasurf.protein.create_surface import (
                    pdb_to_surf_with_min,
                    pdb_to_alpha_complex,
                )

                if self.surface_method == "msms":
                    verts, faces = pdb_to_surf_with_min(pdb_path)
                elif self.surface_method == "alpha_complex":
                    verts, faces = pdb_to_alpha_complex(
                        pdb_path,
                        alpha_value=self.alpha_value,
                        atom_pos=extra_kwargs.get("atom_pos"),
                        atom_radius=extra_kwargs.get("atom_radius"),
                    )
                else:
                    raise ValueError(f"Unknown surface method: {self.surface_method}")

                # DEBUG: Save full surface as PLY
                if self.debug_save_ply:
                    self._save_debug_ply(verts, faces, f"{protein_name}_full")

                # Extract patch from full mesh
                patch_result = self.patch_extractor.extract_patch(
                    verts, faces, pocket_name
                )
                if patch_result is None:
                    print(f"Patch extraction failed for {pocket_name}")
                    return None

                patch_verts, patch_faces = patch_result

                # Apply independent mesh noise BEFORE operator computation
                # This ensures operators (mass, L eigenvalues, etc.) are computed on the noisy mesh
                noise_mode = getattr(self.config, "noise_mode", "none")
                sigma_mesh = getattr(self.config, "sigma_mesh", None)

                if (
                    apply_noise
                    and noise_mode == "independent"
                    and sigma_mesh is not None
                ):
                    import open3d as o3d
                    from alphasurf.protein.create_surface import add_surface_noise

                    # Compute normals for noise direction
                    mesh_temp = o3d.geometry.TriangleMesh()
                    mesh_temp.vertices = o3d.utility.Vector3dVector(patch_verts)
                    mesh_temp.triangles = o3d.utility.Vector3iVector(patch_faces)
                    mesh_temp.compute_vertex_normals()
                    normals = np.asarray(mesh_temp.vertex_normals)

                    # Apply noise
                    patch_verts = add_surface_noise(
                        patch_verts,
                        patch_faces,
                        sigma=sigma_mesh,
                        normals=normals,
                        clip_sigma=getattr(self.config, "clip_sigma", None),
                    )

                # DEBUG: Save patch as PLY
                if self.debug_save_ply:
                    self._save_debug_ply(
                        patch_verts, patch_faces, f"{pocket_name}_patch"
                    )
                    self._debug_count += 1
                    if self._debug_count >= self._debug_max_samples:
                        print(
                            f"[DEBUG] Saved {self._debug_count} samples to {self.debug_ply_dir}"
                        )
                        if self.debug_exit_after_save:
                            print(
                                "[DEBUG] debug_exit_after_save=True, exiting training..."
                            )
                            import sys

                            sys.exit(0)
                        self.debug_save_ply = False

                # Create SurfaceObject from patch mesh WITH operators (only computed on patch!)
                surface = SurfaceObject.from_verts_faces(
                    verts=patch_verts,
                    faces=patch_faces,
                    face_reduction_rate=self.face_reduction_rate,
                    use_pymesh=self.use_pymesh,
                    surface_method=self.surface_method,
                )
                # from_verts_faces computes operators (evals, evecs, etc.)
                # Now add geometric features that depend on operators
                surface.add_geom_feats()
            else:
                # For whole surfaces, generate with operators
                surface = SurfaceObject.from_pdb_path(
                    pdb_path,
                    face_reduction_rate=self.face_reduction_rate,
                    max_vert_number=self.max_vert_number,
                    use_pymesh=self.use_pymesh,
                    surface_method=self.surface_method,
                    alpha_value=self.alpha_value,
                    **extra_kwargs,
                )
                surface.add_geom_feats()

            surface.from_numpy()  # Convert to torch tensors

            # Expand features
            with torch.no_grad():
                surface.expand_features(
                    remove_feats=True, feature_keys=self.feat_keys, oh_keys=self.oh_keys
                )

            # Validate
            if torch.isnan(surface.x).any() or torch.isnan(surface.verts).any():
                print(f"NaN detected in surface for {protein_name}")
                return None

            # Cache if enabled
            if self.use_cache:
                self._cache[protein_name] = surface

            return surface

        except Exception as e:
            print(f"Surface generation failed for {protein_name}: {e}")
            return None

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()


class OnFlyGraphLoader:
    """
    Generates residue/atom graphs on-the-fly from PDB files.
    """

    def __init__(self, config, pdb_dir: str, esm_model_objs=None):
        """
        Args:
            config: Configuration object with graph parameters
            pdb_dir: Path to directory containing PDB files
            esm_model_objs: Optional pre-loaded ESM model tuple (model, batch_converter)
        """
        self.config = config
        self.pdb_dir = pdb_dir

        # Graph parameters
        self.use_graphs = getattr(config, "use_graphs", True)
        self.use_esm = getattr(config, "use_esm", False)
        # Check for precomputed ESM directory (config.esm_dir or default relative path)
        self.esm_dir = getattr(config, "esm_dir", None)
        if self.esm_dir is None:
            # Fallback to default location relative to pdb_dir if not specified
            # pdb_dir is usually .../data/masif_ligand/pdb
            # esm_dir should be .../data/masif_ligand/esm
            potential_esm = os.path.join(os.path.dirname(pdb_dir), "esm")
            if os.path.isdir(potential_esm):
                self.esm_dir = potential_esm

        self.feat_keys = getattr(config, "feat_keys", "all")
        self.oh_keys = getattr(config, "oh_keys", "all")

        # ESM model (lazy load to avoid loading on every worker)
        self._esm_model_objs = esm_model_objs

        # Caching
        self.use_cache = getattr(config, "use_cache", False)
        self._cache: Dict[str, Any] = {}

    def _get_esm_model(self):
        """Lazy load ESM model."""
        if self._esm_model_objs is None and self.use_esm and self.esm_dir is None:
            # Only load model if we are using ESM AND we don't have a precomputed directory
            import esm

            model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            batch_converter = alphabet.get_batch_converter()
            self._esm_model_objs = {
                "model": model,
                "batch_converter": batch_converter,
                "alphabet": alphabet,
                "device": device,
            }
        return self._esm_model_objs

    def _compute_esm_embedding(self, pdb_path: str) -> Optional[torch.Tensor]:
        """Compute ESM embedding for a single PDB."""
        if not self.use_esm:
            return None

        # Check for precomputed embedding first
        if self.esm_dir is not None:
            # Filename format: {pdb_code}_{chain}_esm.pt
            # pdb_path basename: 1ABC_A.pdb -> 1ABC_A
            pdb_basename = os.path.basename(pdb_path)[:-4]
            esm_path = os.path.join(self.esm_dir, f"{pdb_basename}_esm.pt")
            if os.path.exists(esm_path):
                try:
                    return torch.load(esm_path, map_location="cpu")
                except Exception as e:
                    print(f"Failed to load precomputed ESM for {pdb_basename}: {e}")
                    # Fall through to computation if load fails

        try:
            from alphasurf.protein.graphs import quick_pdb_to_seq, res_type_idx_to_1

            model_objs = self._get_esm_model()
            if model_objs is None:
                # If model is not loaded (maybe because we expected precomputed), we can't compute
                return None

            model = model_objs["model"]
            batch_converter = model_objs["batch_converter"]
            device = model_objs["device"]

            # Get sequence
            seq = quick_pdb_to_seq(pdb_path)
            seq = "".join([res_type_idx_to_1[i] for i in seq])
            name = os.path.basename(pdb_path)[:-4]

            # Compute embedding
            batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33])

            embed = results["representations"][33][:, 1:-1, :][0]
            return embed.cpu()

        except Exception as e:
            print(f"ESM embedding failed for {pdb_path}: {e}")
            return None

    def load(
        self, pocket_name: str, parsed_arrays: Optional[tuple] = None
    ) -> Optional[Data]:
        """
        Generate or retrieve a graph for the given pocket/protein.

        Args:
            pocket_name: Name like "1ABC_A_patch_0_HEM"
            parsed_arrays: Optional pre-parsed PDB arrays from OnFlyPDBParser.
                          Avoids redundant parsing when shared with surface loader.

        Returns:
            ResidueGraph with features expanded, or None on failure
        """
        if not self.use_graphs:
            return Data()

        # Get protein name (always use protein, not patch)
        protein_name = pocket_name.split("_patch_")[0]

        # Check cache
        if self.use_cache and protein_name in self._cache:
            return self._cache[protein_name]

        pdb_path = os.path.join(self.pdb_dir, f"{protein_name}.pdb")
        if not os.path.exists(pdb_path):
            print(f"PDB file not found: {pdb_path}")
            return None

        try:
            # Use pre-parsed arrays if available, otherwise parse
            if parsed_arrays is not None:
                arrays = parsed_arrays
            else:
                try:
                    arrays = parse_pdb_path(pdb_path, use_pqr=False)
                except:
                    arrays = parse_pdb_path(pdb_path)

            # Build residue graph
            graph = ResidueGraphBuilder(
                add_pronet=True, add_esm=False
            ).arrays_to_resgraph(arrays)

            # Add ESM features if requested
            if self.use_esm:
                esm_feats = self._compute_esm_embedding(pdb_path)
                if esm_feats is not None:
                    graph.features.add_named_features("esm_feats", esm_feats)
                    if self.feat_keys != "all":
                        self.feat_keys = list(self.feat_keys) + ["esm_feats"]

            # Patch for node_len if missing
            if "node_len" not in graph.keys():
                graph.node_len = len(graph.node_pos)

            # Expand features
            with torch.no_grad():
                graph.expand_features(
                    remove_feats=True, feature_keys=self.feat_keys, oh_keys=self.oh_keys
                )

            # Validate
            if torch.isnan(graph.x).any() or torch.isnan(graph.node_pos).any():
                print(f"NaN detected in graph for {protein_name}")
                return None

            # Cache if enabled
            if self.use_cache:
                self._cache[protein_name] = graph

            return graph

        except Exception as e:
            print(f"Graph generation failed for {protein_name}: {e}")
            return None

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()


class MasifLigandDatasetOnFly(Dataset):
    """
    Dataset that generates surfaces and graphs on-the-fly.

    Supports noise augmentation modes:
    - 'none': No augmentation (default)
    - 'independent': Noise graph coordinates and mesh vertices independently
    - 'joint': Noise atom coordinates, regenerate mesh from noised positions
    """

    def __init__(
        self,
        systems: Dict,
        surface_loader: OnFlySurfaceLoader,
        graph_loader: OnFlyGraphLoader,
        pdb_parser: Optional[OnFlyPDBParser] = None,
        use_shared_parsing: bool = False,
        noise_augmentor: Optional[NoiseAugmentor] = None,
    ):
        """
        Args:
            systems: Dictionary of {pocket_name: (lig_coord, lig_type)}
            surface_loader: OnFlySurfaceLoader instance
            graph_loader: OnFlyGraphLoader instance
            pdb_parser: Optional shared PDB parser for coordinated parsing
            use_shared_parsing: If True and pdb_parser is provided, parse PDB once
                               and share arrays between surface and graph loaders.
                               Most beneficial when using alpha_complex method.
            noise_augmentor: Optional NoiseAugmentor for data augmentation
        """
        self.systems = systems
        self.systems_keys = list(systems.keys())
        self.surface_loader = surface_loader
        self.graph_loader = graph_loader
        self.pdb_parser = pdb_parser
        self.use_shared_parsing = use_shared_parsing and (pdb_parser is not None)
        self.noise_augmentor = noise_augmentor

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        pocket = self.systems_keys[idx]
        lig_coord, lig_type = self.systems[pocket]
        lig_coord = torch.from_numpy(lig_coord)

        # Determine if we need fresh data (no caching) for noise augmentation
        # For joint mode, we need to parse fresh and apply noise before generation
        # For independent mode, we can use cached data and noise after
        use_noise = (
            self.noise_augmentor is not None
            and self.noise_augmentor.noise_mode != "none"
        )
        is_joint_mode = use_noise and self.noise_augmentor.noise_mode == "joint"

        # Parse PDB for shared parsing or joint noise mode
        parsed_arrays = None
        if self.use_shared_parsing or is_joint_mode:
            protein_name = pocket.split("_patch_")[0]
            if self.pdb_parser is not None:
                parsed_arrays = self.pdb_parser.parse(protein_name)
            else:
                # Parse directly if no shared parser
                pdb_path = self.surface_loader.pdb_dir
                pdb_file = os.path.join(pdb_path, f"{protein_name}.pdb")
                try:
                    parsed_arrays = parse_pdb_path(pdb_file, use_pqr=False)
                except:
                    parsed_arrays = parse_pdb_path(pdb_file)
            if parsed_arrays is None:
                return None

        # For joint mode: noise the parsed arrays BEFORE generating surface/graph
        if is_joint_mode and parsed_arrays is not None:
            parsed_arrays = self.noise_augmentor.noise_parsed_arrays(parsed_arrays)

        # Generate surface and graph (from noised arrays if joint mode)
        surface = self.surface_loader.load(
            pocket, parsed_arrays=parsed_arrays, apply_noise=use_noise
        )
        graph = self.graph_loader.load(pocket, parsed_arrays=parsed_arrays)

        if surface is None or graph is None:
            return None

        # For independent mode: noise the graph (mesh is already noised inside surface_loader)
        if use_noise and self.noise_augmentor.noise_mode == "independent":
            self.noise_augmentor.noise_graph(graph)

        # Validate for NaN
        if (
            hasattr(surface, "x")
            and surface.x is not None
            and torch.isnan(surface.x).any()
        ) or (
            hasattr(graph, "x") and graph.x is not None and torch.isnan(graph.x).any()
        ):
            return None

        item = Data(surface=surface, graph=graph, lig_coord=lig_coord, label=lig_type)
        return item


class MasifLigandDataModuleOnFly(pl.LightningDataModule):
    """
    Lightning DataModule that generates all data on-the-fly.

    This eliminates the need for a separate preprocessing step.
    Supports noise augmentation for training data.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Determine data directory
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if cfg.data_dir is None:
            masif_ligand_data_dir = os.path.join(
                script_dir, "..", "..", "..", "data", "masif_ligand"
            )
        else:
            masif_ligand_data_dir = cfg.data_dir

        # PDB directory
        self.pdb_dir = os.path.join(
            masif_ligand_data_dir, "raw_data_MasifLigand", "pdb"
        )

        # Patch directory (for reference patch vertices)
        self.patch_dir = os.path.join(masif_ligand_data_dir, "dataset_MasifLigand")

        # Create on-fly loaders with config from cfg_surface/cfg_graph + on_fly overrides
        on_fly_cfg = getattr(cfg, "on_fly", None)

        # Check if we should use whole surfaces or extract patches
        use_whole_surfaces = (
            getattr(on_fly_cfg, "use_whole_surfaces", True) if on_fly_cfg else True
        )
        patch_radius = getattr(on_fly_cfg, "patch_radius", 6.0) if on_fly_cfg else 6.0
        min_faces = getattr(on_fly_cfg, "min_faces", 250) if on_fly_cfg else 250
        patch_max_radius = (
            getattr(on_fly_cfg, "patch_max_radius", 12.0) if on_fly_cfg else 12.0
        )

        # Create patch extractor if not using whole surfaces
        if not use_whole_surfaces and os.path.exists(self.patch_dir):
            min_verts = getattr(on_fly_cfg, "min_verts", 140) if on_fly_cfg else 140
            patch_extractor = PatchExtractor(
                self.patch_dir,
                radius=patch_radius,
                min_verts=min_verts,
                max_radius=patch_max_radius,
            )
            print(
                f"[PATCH] Extracting patches with radius {patch_radius}Å (min_verts={min_verts}, max_radius={patch_max_radius}Å) from {self.patch_dir}"
            )
        else:
            patch_extractor = None
            if not use_whole_surfaces:
                print(
                    f"[WARNING] Patch dir not found: {self.patch_dir}, using whole surfaces"
                )

        # Build surface config
        surface_config = self._merge_configs(cfg.cfg_surface, on_fly_cfg)
        self.surface_loader = OnFlySurfaceLoader(
            surface_config, self.pdb_dir, patch_extractor=patch_extractor
        )

        # Build graph config
        graph_config = self._merge_configs(cfg.cfg_graph, on_fly_cfg)
        self.graph_loader = OnFlyGraphLoader(graph_config, self.pdb_dir)

        # Create shared PDB parser for alpha_complex method or joint noise mode
        # Joint noise mode needs fresh parsed arrays to noise before generation
        noise_mode = getattr(on_fly_cfg, "noise_mode", "none") if on_fly_cfg else "none"
        self.use_shared_parsing = (
            self.surface_loader.surface_method == "alpha_complex"
            or noise_mode == "joint"
        )
        use_cache = getattr(on_fly_cfg, "use_cache", False) if on_fly_cfg else True
        # For joint mode with noise, don't cache parsed arrays (need fresh each time)
        if noise_mode == "joint":
            use_cache = False
        if self.use_shared_parsing:
            self.pdb_parser = OnFlyPDBParser(self.pdb_dir, use_cache=use_cache)
        else:
            self.pdb_parser = None

        # Create noise augmentor for training (only applied during training)
        if on_fly_cfg is not None and noise_mode != "none":
            sigma_graph = getattr(on_fly_cfg, "sigma_graph", 0.3)
            sigma_mesh = getattr(on_fly_cfg, "sigma_mesh", 0.3)
            clip_sigma = getattr(on_fly_cfg, "clip_sigma", 3.0)
            self.noise_augmentor = NoiseAugmentor(
                noise_mode=noise_mode,
                sigma_graph=sigma_graph,
                sigma_mesh=sigma_mesh,
                clip_sigma=clip_sigma,
            )
            print(
                f"[NOISE] Mode: {noise_mode}, sigma_graph: {sigma_graph}, sigma_mesh: {sigma_mesh}"
            )
        else:
            self.noise_augmentor = None

        # Load split systems
        splits_dir = os.path.join(
            masif_ligand_data_dir, "raw_data_MasifLigand", "splits"
        )
        ligands_path = os.path.join(
            masif_ligand_data_dir, "raw_data_MasifLigand", "ligand"
        )

        self.systems = []
        for split in ["train", "val", "test"]:
            splits_path = os.path.join(splits_dir, f"{split}-list.txt")
            out_path = os.path.join(splits_dir, f"{split}.p")
            self.systems.append(
                get_systems_from_ligands(
                    splits_path, ligands_path=ligands_path, out_path=out_path
                )
            )

        # DataLoader arguments
        prefetch_factor = getattr(self.cfg.loader, "prefetch_factor", None)
        persistent_workers = getattr(self.cfg.loader, "persistent_workers", False)
        self.loader_args = {
            "num_workers": self.cfg.loader.num_workers,
            "batch_size": self.cfg.loader.batch_size,
            "pin_memory": self.cfg.loader.pin_memory,
            "drop_last": self.cfg.loader.drop_last,
            "collate_fn": self._collate_fn,
        }
        if self.cfg.loader.num_workers and self.cfg.loader.num_workers > 0:
            if prefetch_factor is not None:
                self.loader_args["prefetch_factor"] = prefetch_factor
            self.loader_args["persistent_workers"] = persistent_workers

        # Update model input dim (without noise augmentor for dimension calculation)
        dataset_temp = MasifLigandDatasetOnFly(
            self.systems[0],
            self.surface_loader,
            self.graph_loader,
            pdb_parser=self.pdb_parser,
            use_shared_parsing=self.use_shared_parsing,
        )
        update_model_input_dim(cfg, dataset_temp=dataset_temp)

    def _merge_configs(self, base_config, override_config):
        """Merge base config with on_fly overrides."""
        if override_config is None:
            return base_config

        # Create a merged config
        merged = Data()
        for key in dir(base_config):
            if not key.startswith("_"):
                try:
                    setattr(merged, key, getattr(base_config, key))
                except:
                    pass

        # Apply overrides
        for key in dir(override_config):
            if not key.startswith("_"):
                try:
                    setattr(merged, key, getattr(override_config, key))
                except:
                    pass

        return merged

    @staticmethod
    def _collate_fn(x):
        return AtomBatch.from_data_list(x)

    def train_dataloader(self):
        # Noise augmentation only for training
        dataset = MasifLigandDatasetOnFly(
            self.systems[0],
            self.surface_loader,
            self.graph_loader,
            pdb_parser=self.pdb_parser,
            use_shared_parsing=self.use_shared_parsing,
            noise_augmentor=self.noise_augmentor,  # Only train gets noise
        )
        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def val_dataloader(self):
        dataset = MasifLigandDatasetOnFly(
            self.systems[1],
            self.surface_loader,
            self.graph_loader,
            pdb_parser=self.pdb_parser,
            use_shared_parsing=self.use_shared_parsing,
        )
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        dataset = MasifLigandDatasetOnFly(
            self.systems[2],
            self.surface_loader,
            self.graph_loader,
            pdb_parser=self.pdb_parser,
            use_shared_parsing=self.use_shared_parsing,
        )
        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == "__main__":
    # Quick test
    from torch_geometric.data import Data as TGData

    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(
        script_dir, "..", "..", "..", "data", "masif_ligand", "msms_01"
    )
    pdb_dir = os.path.join(masif_ligand_data_dir, "raw_data_MasifLigand", "pdb")

    # Test surface loader
    cfg_surface = TGData()
    cfg_surface.use_surfaces = True
    cfg_surface.surface_method = "msms"
    cfg_surface.face_reduction_rate = 0.1
    cfg_surface.feat_keys = "all"
    cfg_surface.oh_keys = "all"
    cfg_surface.use_whole_surfaces = True
    cfg_surface.use_cache = True

    surface_loader = OnFlySurfaceLoader(cfg_surface, pdb_dir)
    surface = surface_loader.load("1A27_AB_patch_0_HEM")
    if surface is not None:
        print(f"Surface loaded: {surface.verts.shape[0]} vertices")
    else:
        print("Surface loading failed")
