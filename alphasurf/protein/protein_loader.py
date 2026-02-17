"""
Unified ProteinLoader supporting both disk and on-the-fly modes.

This module consolidates surface and graph loading with integrated transform
support. Transforms (noise, patch extraction) are applied DURING generation
so that computed features (operators, edges, etc.) reflect the transformed geometry.

Transform Order:
1. Parse PDB -> raw arrays
2. Apply atom noise (joint: both, independent: graph only)
3. Generate mesh from atoms
4. Extract patch (if configured)
5. Apply mesh noise (independent mode only)
6. Compute operators on final mesh
7. Build graph from (noised) arrays
8. Expand features
"""

import logging
import os
from typing import Any, Literal, Optional, Tuple

import numpy as np
import torch
from alphasurf.protein.graphs import parse_pdb_path
from alphasurf.protein.protein import Protein
from alphasurf.protein.residue_graph import ResidueGraphBuilder
from alphasurf.protein.surfaces import SurfaceObject
from alphasurf.protein.transforms import NoiseAugmentor, PatchExtractor, add_mesh_noise
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class ProteinLoader:
    """
    Load proteins either from disk or generate on-the-fly.

    This is the unified entry point for protein loading. Transforms (noise,
    patch extraction) are applied during generation, ensuring that all
    computed features (operators, edges, etc.) reflect the transformed geometry.

    Modes:
        - "disk": Load precomputed surfaces and graphs from directory
        - "on_fly": Generate surfaces and graphs from PDB files

    Example:
        >>> loader = ProteinLoader(
        ...     mode="on_fly",
        ...     pdb_dir="/path/to/pdb",
        ...     surface_config=surface_cfg,
        ...     graph_config=graph_cfg,
        ... )
        >>> protein = loader.load("1ABC_A")
    """

    def __init__(
        self,
        mode: Literal["disk", "on_fly"] = "on_fly",
        # Directory paths
        pdb_dir: Optional[str] = None,
        surface_dir: Optional[str] = None,
        graph_dir: Optional[str] = None,
        esm_dir: Optional[str] = None,
        # Configs
        surface_config: Optional[Any] = None,
        graph_config: Optional[Any] = None,
        # Transform options
        noise_augmentor: Optional[NoiseAugmentor] = None,
        patch_extractor: Optional[PatchExtractor] = None,
    ):
        """
        Args:
            mode: "disk" to load from preprocessed files, "on_fly" to generate
            pdb_dir: Directory containing PDB files (required for on_fly)
            surface_dir: Directory with precomputed surfaces (for disk mode)
            graph_dir: Directory with precomputed graphs (for disk mode)
            esm_dir: Directory with precomputed ESM embeddings
            surface_config: Configuration for surface generation
            graph_config: Configuration for graph generation
            noise_augmentor: NoiseAugmentor instance for coordinate noise
            patch_extractor: PatchExtractor instance for binding site extraction
        """
        self.mode = mode
        self.pdb_dir = pdb_dir
        self.surface_dir = surface_dir
        self.graph_dir = graph_dir
        self.esm_dir = esm_dir
        self.surface_config = surface_config
        self.graph_config = graph_config
        self.noise_augmentor = noise_augmentor
        self.patch_extractor = patch_extractor

        if mode == "on_fly" and pdb_dir is None:
            raise ValueError("pdb_dir is required for on_fly mode")
        if mode == "disk" and surface_dir is None and graph_dir is None:
            raise ValueError(
                "At least one of surface_dir, graph_dir is required for disk mode"
            )

        # Extract feature keys once (avoid mutation during load)
        self._surface_feat_keys = self._get_feat_keys(surface_config)
        self._surface_oh_keys = self._get_oh_keys(surface_config)
        self._graph_feat_keys = self._get_feat_keys(graph_config)
        self._graph_oh_keys = self._get_oh_keys(graph_config)

    @staticmethod
    def _get_feat_keys(config) -> Any:
        if config is None:
            return "all"
        return getattr(config, "feat_keys", "all")

    @staticmethod
    def _get_oh_keys(config) -> Any:
        if config is None:
            return "all"
        return getattr(config, "oh_keys", "all")

    def load(
        self,
        name: str,
        pdb_path: Optional[str] = None,
        apply_noise: bool = False,
    ) -> Optional[Protein]:
        """
        Load or generate a Protein.

        Transforms are applied DURING generation, ensuring features reflect
        the transformed geometry.

        Args:
            name: Protein identifier (e.g., "1ABC_A" or "1ABC_A_patch_0_HEM")
            pdb_path: explicit path to PDB file (overrides pdb_dir/name.pdb)
            apply_noise: If True, apply noise augmentation during generation

        Returns:
            Protein object with surface and graph, or None on failure
        """
        if self.mode == "disk":
            protein = self._load_from_disk(name)
        else:
            protein = self._generate_on_fly(
                name, pdb_path=pdb_path, apply_noise=apply_noise
            )

        if protein is None:
            return None

        if not protein.validate():
            return None

        return protein

    def _load_from_disk(self, name: str) -> Optional[Protein]:
        """Load precomputed surface and graph from disk."""
        if "_patch_" in name:
            protein_name = name.split("_patch_")[0]
            pocket_name = name
        else:
            protein_name = name
            pocket_name = None

        surface = None
        graph = None

        if self.surface_dir is not None:
            surface_key = pocket_name or protein_name
            surface_path = os.path.join(self.surface_dir, f"{surface_key}.pt")
            if os.path.exists(surface_path):
                try:
                    surface = torch.load(surface_path, weights_only=False)
                except Exception as e:
                    logger.warning("Failed to load surface for %s: %s", surface_key, e)

        if self.graph_dir is not None:
            graph_path = os.path.join(self.graph_dir, f"{protein_name}.pt")
            if os.path.exists(graph_path):
                try:
                    graph = torch.load(graph_path, weights_only=False)
                except Exception as e:
                    logger.warning("Failed to load graph for %s: %s", protein_name, e)

        if surface is None and graph is None:
            return None

        # Expand features (critical for creating .x attribute for batching)
        if surface is not None:
            with torch.no_grad():
                surface.expand_features(
                    remove_feats=True,
                    feature_keys=self._surface_feat_keys,
                    oh_keys=self._surface_oh_keys,
                )

        if graph is not None:
            if "node_len" not in graph.keys():
                graph.node_len = len(graph.node_pos)

            # Load ESM if configured
            use_esm = (
                getattr(self.graph_config, "use_esm", False)
                if self.graph_config
                else False
            )
            if use_esm:
                esm_feats = self._load_esm_embedding(protein_name)
                if esm_feats is not None:
                    graph.features.add_named_features("esm_feats", esm_feats)

            with torch.no_grad():
                # Determine keys to use
                feat_keys = self._graph_feat_keys
                if use_esm and feat_keys != "all" and esm_feats is not None:
                    feat_keys = list(feat_keys) + ["esm_feats"]

                graph.expand_features(
                    remove_feats=True,
                    feature_keys=feat_keys,
                    oh_keys=self._graph_oh_keys,
                )

        # Populate metadata from graph if available (for disk mode full-atom interface)
        metadata = {}
        if graph is not None:
            if hasattr(graph, "atom_pos"):
                metadata["atom_pos"] = graph.atom_pos
                del graph.atom_pos
            if hasattr(graph, "atom_res_map"):
                metadata["atom_res_map"] = graph.atom_res_map
                del graph.atom_res_map

        return Protein(
            surface=surface,
            graph=graph,
            name=protein_name,
            pdb_path=os.path.join(self.pdb_dir, f"{protein_name}.pdb")
            if self.pdb_dir
            else None,
            metadata=metadata,
        )

    def _generate_on_fly(
        self,
        name: str,
        pdb_path: Optional[str] = None,
        apply_noise: bool = False,
    ) -> Optional[Protein]:
        """Generate surface and graph on-the-fly from PDB."""
        if "_patch_" in name:
            protein_name = name.split("_patch_")[0]
            pocket_name = name
        else:
            protein_name = name
            pocket_name = None

        if pdb_path is None:
            pdb_path = os.path.join(self.pdb_dir, f"{protein_name}.pdb")

        if not os.path.exists(pdb_path):
            logger.warning("PDB file not found: %s", pdb_path)
            return None

        parsed_arrays = self._parse_pdb(pdb_path)
        if parsed_arrays is None:
            return None

        # Determine noise mode
        use_noise = (
            apply_noise
            and self.noise_augmentor is not None
            and self.noise_augmentor.enabled
        )
        noise_mode = self.noise_augmentor.mode if use_noise else "none"

        # Prepare arrays for surface vs graph
        # Joint: same noised arrays for both
        # Independent: graph gets noised arrays, surface gets original (mesh noise later)
        parsed_for_surface = parsed_arrays
        parsed_for_graph = parsed_arrays

        if noise_mode == "joint":
            noised = self.noise_augmentor.noise_arrays(parsed_arrays)
            parsed_for_surface = noised
            parsed_for_graph = noised
        elif noise_mode == "independent":
            parsed_for_graph = self.noise_augmentor.noise_arrays(parsed_arrays)

        # Sample random alpha value if alpha noise mode
        alpha_override = None
        if noise_mode == "alpha":
            alpha_override = self.noise_augmentor.sample_alpha_value()

        # Generate surface
        surface = self._generate_surface(
            pdb_path=pdb_path,
            protein_name=protein_name,
            pocket_name=pocket_name,
            parsed_arrays=parsed_for_surface,
            apply_mesh_noise=(noise_mode == "independent"),
            alpha_override=alpha_override,
        )

        # Generate graph from (potentially noised) arrays
        graph = self._generate_graph(
            pdb_path=pdb_path,
            protein_name=protein_name,
            parsed_arrays=parsed_for_graph,
        )

        if surface is None and graph is None:
            return None

        # Store atom-level data for interface computation
        metadata = {}
        if parsed_arrays is not None:
            # Extract atom positions and residue mapping
            # arrays format: amino_types, atom_chain_id, atom_amino_id, atom_names,
            #                atom_types, atom_pos, atom_charge, atom_radius, res_sse, amino_ids, atom_ids
            atom_amino_id = parsed_arrays[2]  # residue index for each atom
            atom_pos = parsed_arrays[5]  # atom positions
            metadata["atom_pos"] = torch.from_numpy(atom_pos).float()
            metadata["atom_res_map"] = torch.from_numpy(atom_amino_id).long()

        return Protein(
            surface=surface,
            graph=graph,
            name=protein_name,
            pdb_path=pdb_path,
            metadata=metadata,
        )

    def _parse_pdb(self, pdb_path: str) -> Optional[Tuple]:
        """Parse PDB file and ensure float32 for critical arrays."""
        try:
            try:
                arrays = parse_pdb_path(pdb_path, use_pqr=False)
            except TypeError:
                arrays = parse_pdb_path(pdb_path)

            # Ensure float32 for consistency
            arrays_list = list(arrays)
            if len(arrays_list) > 7:
                if arrays_list[5].dtype != np.float32:
                    arrays_list[5] = arrays_list[5].astype(np.float32)
                if arrays_list[7].dtype != np.float32:
                    arrays_list[7] = arrays_list[7].astype(np.float32)
            return tuple(arrays_list)

        except Exception as e:
            logger.warning("PDB parsing failed for %s: %s", pdb_path, e)
            return None

    def _generate_surface(
        self,
        pdb_path: str,
        protein_name: str,
        pocket_name: Optional[str],
        parsed_arrays: Tuple,
        apply_mesh_noise: bool = False,
        alpha_override: Optional[float] = None,
    ) -> Optional[SurfaceObject]:
        """Generate surface with optional patch extraction and mesh noise."""
        if self.surface_config is None:
            return None

        cfg = self.surface_config
        if not getattr(cfg, "use_surfaces", True):
            return Data()

        surface_method = getattr(cfg, "surface_method", "msms")
        face_reduction_rate = getattr(cfg, "face_reduction_rate", 0.1)
        alpha_value = getattr(cfg, "alpha_value", 0.1)
        # Use random alpha if provided (alpha noise augmentation)
        if alpha_override is not None:
            alpha_value = alpha_override
        min_vert_number = getattr(cfg, "min_vert_number", 16)
        use_pymesh = getattr(cfg, "use_pymesh", False)
        use_whole_surfaces = getattr(cfg, "use_whole_surfaces", True)
        precomputed_patches_dir = getattr(cfg, "precomputed_patches_dir", None)

        try:
            extra_kwargs = {}
            if surface_method == "alpha_complex" and parsed_arrays is not None:
                extra_kwargs["atom_pos"] = parsed_arrays[5]
                extra_kwargs["atom_radius"] = parsed_arrays[7]

            uses_precomputed_msms = (
                surface_method == "msms" and precomputed_patches_dir is not None
            )
            should_extract_patch = not use_whole_surfaces and (
                pocket_name is not None or uses_precomputed_msms
            )

            print(f"DEBUG: Starting surface gen for {protein_name}")
            if should_extract_patch:
                # ... (omitted) ...
                if uses_precomputed_msms and pocket_name:
                    patch_path = os.path.join(
                        precomputed_patches_dir, f"{pocket_name}.pt"
                    )
                    if os.path.exists(patch_path):
                        patch_data = torch.load(patch_path, weights_only=False)
                        patch_verts = np.asarray(patch_data.verts)
                        patch_faces = np.asarray(patch_data.faces)
                    else:
                        logger.warning("Precomputed patch not found: %s", patch_path)
                        return None
                else:
                    from alphasurf.protein.create_surface import (
                        pdb_to_alpha_complex,
                        pdb_to_surf_with_min,
                    )

                    if surface_method == "msms":
                        verts, faces = pdb_to_surf_with_min(
                            pdb_path, min_number=min_vert_number
                        )
                    elif surface_method == "alpha_complex":
                        verts, faces, _, _ = pdb_to_alpha_complex(
                            pdb_path,
                            alpha_value=alpha_value,
                            atom_pos=extra_kwargs.get("atom_pos"),
                            atom_radius=extra_kwargs.get("atom_radius"),
                        )
                    else:
                        raise ValueError(f"Unknown surface method: {surface_method}")

                    if self.patch_extractor is not None and pocket_name is not None:
                        result = self.patch_extractor.extract_patch(
                            verts, faces, pocket_name
                        )
                        if result is None:
                            logger.warning(
                                "Patch extraction failed for %s", pocket_name
                            )
                            return None
                        patch_verts, patch_faces = result
                    else:
                        patch_verts, patch_faces = verts, faces

                # Apply mesh noise BEFORE computing operators (independent mode)
                if apply_mesh_noise and self.noise_augmentor is not None:
                    # Note: patch_verts/faces are just verts/faces if no extraction happened
                    patch_verts = add_mesh_noise(
                        patch_verts,
                        patch_faces,
                        sigma=self.noise_augmentor.sigma_mesh,
                        clip_sigma=self.noise_augmentor.clip_sigma,
                    )

                surface = SurfaceObject.from_verts_faces(
                    verts=patch_verts,
                    faces=patch_faces,
                    face_reduction_rate=face_reduction_rate,
                    use_pymesh=use_pymesh,
                    surface_method=surface_method,
                    min_vert_number=min_vert_number,
                )

                surface.add_geom_feats()
            else:
                from alphasurf.protein.create_surface import (
                    pdb_to_alpha_complex,
                    pdb_to_surf_with_min,
                )

                if surface_method == "msms":
                    verts, faces = pdb_to_surf_with_min(
                        pdb_path, min_number=min_vert_number
                    )
                elif surface_method == "alpha_complex":
                    verts, faces, _, _ = pdb_to_alpha_complex(
                        pdb_path,
                        alpha_value=alpha_value,
                        atom_pos=extra_kwargs.get("atom_pos"),
                        atom_radius=extra_kwargs.get("atom_radius"),
                    )
                else:
                    raise ValueError(f"Unknown surface method: {surface_method}")

                # Apply mesh noise BEFORE computing operators
                if apply_mesh_noise and self.noise_augmentor is not None:
                    verts = add_mesh_noise(
                        verts,
                        faces,
                        sigma=self.noise_augmentor.sigma_mesh,
                        clip_sigma=self.noise_augmentor.clip_sigma,
                    )

                surface = SurfaceObject.from_verts_faces(
                    verts=verts,
                    faces=faces,
                    face_reduction_rate=face_reduction_rate,
                    use_pymesh=use_pymesh,
                    surface_method=surface_method,
                    min_vert_number=min_vert_number,
                )

                surface.add_geom_feats()

            surface.from_numpy()

            with torch.no_grad():
                surface.expand_features(
                    remove_feats=True,
                    feature_keys=self._surface_feat_keys,
                    oh_keys=self._surface_oh_keys,
                )

            return surface

        except Exception as e:
            logger.warning("Surface generation failed for %s: %s", protein_name, e)

            return None

    def _generate_graph(
        self,
        pdb_path: str,
        protein_name: str,
        parsed_arrays: Tuple,
    ) -> Optional[Data]:
        """Generate residue graph from parsed arrays."""
        if self.graph_config is None:
            return None

        cfg = self.graph_config
        if not getattr(cfg, "use_graphs", True):
            return Data()

        use_esm = getattr(cfg, "use_esm", False)

        try:
            graph = ResidueGraphBuilder(
                add_pronet=True, add_esm=False
            ).arrays_to_resgraph(parsed_arrays)

            # Determine feat_keys (copy to avoid mutation)
            feat_keys = self._graph_feat_keys
            if feat_keys != "all":
                feat_keys = list(feat_keys)

            if use_esm:
                esm_feats = self._load_esm_embedding(protein_name)
                if esm_feats is None:
                    raise RuntimeError(
                        f"ESM embedding not found for {protein_name}. "
                        f"use_esm=True requires precomputed ESM in {self.esm_dir}."
                    )
                graph.features.add_named_features("esm_feats", esm_feats)
                if feat_keys != "all":
                    feat_keys = feat_keys + ["esm_feats"]

            if "node_len" not in graph.keys():
                graph.node_len = len(graph.node_pos)

            with torch.no_grad():
                graph.expand_features(
                    remove_feats=True,
                    feature_keys=feat_keys,
                    oh_keys=self._graph_oh_keys,
                )

            return graph

        except Exception as e:
            logger.warning("Graph generation failed for %s: %s", protein_name, e)
            return None

    def _load_esm_embedding(self, protein_name: str) -> Optional[torch.Tensor]:
        """Load precomputed ESM embedding from disk."""
        if self.esm_dir is None:
            return None

        esm_path = os.path.join(self.esm_dir, f"{protein_name}_esm.pt")
        if os.path.exists(esm_path):
            try:
                return torch.load(esm_path, map_location="cpu")
            except Exception as e:
                logger.warning("Failed to load ESM for %s: %s", protein_name, e)

        return None
