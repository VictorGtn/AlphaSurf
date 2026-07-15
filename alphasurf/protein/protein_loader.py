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
from alphasurf.protein.transforms import NoiseAugmentor, PatchExtractor
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "CB"}


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
        crop_window: Optional[Tuple[int, int]] = None,
        ala_strip_positions: Optional[list] = None,
        ala_strip_keep_cb: bool = True,
    ) -> Optional[Protein]:
        """
        Load or generate a Protein.

        Transforms are applied DURING generation, ensuring features reflect
        the transformed geometry. Noise is applied iff self.noise_augmentor is
        not None and enabled.

        Args:
            name: Protein identifier (e.g., "1ABC_A" or "1ABC_A_patch_0_HEM")
            pdb_path: explicit path to PDB file (overrides pdb_dir/name.pdb)
            crop_window: if set (on_fly mode only), crop residues [start, end)
                BEFORE surface/graph generation. Surface is built from cropped
                atoms only — no mesh-operator rebuild needed.
            ala_strip_positions: if set (on_fly mode only), replace sidechain
                atoms (beyond Cb) with nothing at these residue indices — only
                N, CA, C, O, CB remain. Applied after crop, before surface/graph
                generation. Used for S3F-style masked-residue leakage prevention.
            ala_strip_keep_cb: retain CB at stripped positions. Disable for a
                uniform N/CA/C/O backbone mask (including glycine).

        Returns:
            Protein object with surface and graph, or None on failure
        """
        if self.mode == "disk":
            protein = self._load_from_disk(name)
        else:
            protein = self._generate_on_fly(
                name,
                pdb_path=pdb_path,
                crop_window=crop_window,
                ala_strip_positions=ala_strip_positions,
                ala_strip_keep_cb=ala_strip_keep_cb,
            )

        if protein is None:
            return None

        if not protein.validate():
            return None

        return protein

    def load_clean(
        self,
        name: str,
        pdb_path: Optional[str] = None,
    ) -> Optional[Protein]:
        """Load a Protein with noise augmentation temporarily disabled.

        Used by cache pre-population paths that must always load clean geometry
        regardless of the loader's configured augmentor.
        """
        saved = self.noise_augmentor
        self.noise_augmentor = None
        try:
            return self.load(name, pdb_path=pdb_path)
        finally:
            self.noise_augmentor = saved

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
            if hasattr(surface, "features") and surface.features is not None:
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
            if use_esm and hasattr(graph, "features") and graph.features is not None:
                esm_feats = self._load_esm_embedding(protein_name)
                if esm_feats is not None:
                    graph.features.add_named_features("esm_feats", esm_feats)

            if hasattr(graph, "features") and graph.features is not None:
                with torch.no_grad():
                    # Determine keys to use
                    feat_keys = self._graph_feat_keys
                    if (
                        use_esm
                        and feat_keys != "all"
                        and hasattr(graph, "features")
                        and esm_feats is not None
                    ):
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
        crop_window: Optional[Tuple[int, int]] = None,
        ala_strip_positions: Optional[list] = None,
        ala_strip_keep_cb: bool = True,
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

        if crop_window is not None:
            parsed_arrays = self._crop_parsed_arrays(parsed_arrays, *crop_window)

        if ala_strip_positions:
            parsed_arrays = self._strip_sidechains_to_ala(
                parsed_arrays,
                ala_strip_positions,
                keep_cb=ala_strip_keep_cb,
            )

        if self.noise_augmentor is not None and self.noise_augmentor.enabled:
            (
                parsed_for_surface,
                parsed_for_graph,
                alpha_override,
            ) = self.noise_augmentor.prepare_arrays(parsed_arrays)
        else:
            parsed_for_surface = parsed_arrays
            parsed_for_graph = parsed_arrays
            alpha_override = None

        # Generate surface
        surface = self._generate_surface(
            pdb_path=pdb_path,
            protein_name=protein_name,
            pocket_name=pocket_name,
            parsed_arrays=parsed_for_surface,
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

        # Store atom-level data for interface computation.
        # Use parsed_for_graph so metadata reflects the same atoms the graph saw
        # (noised under joint/independent modes, clean otherwise).
        metadata = {}
        if parsed_for_graph is not None:
            atom_amino_id = parsed_for_graph[2]
            atom_pos = parsed_for_graph[5]
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

    @staticmethod
    def _crop_parsed_arrays(arrays: Tuple, start: int, end: int) -> Tuple:
        """Crop parsed PDB arrays to residues [start, end).

        Residue-level arrays (amino_types, res_sse, amino_ids) are sliced.
        Atom-level arrays are filtered to atoms whose residue index is in
        [start, end), and atom_amino_id is remapped to 0-indexed within the
        crop. This lets surface/graph generation run on the cropped structure
        exactly as if it were a standalone PDB.
        """
        (
            amino_types,
            atom_chain_id,
            atom_amino_id,
            atom_names,
            atom_types,
            atom_pos,
            atom_charge,
            atom_radius,
            res_sse,
            amino_ids,
            atom_ids,
        ) = arrays

        amino_types = amino_types[start:end]
        res_sse = res_sse[start:end]
        amino_ids = amino_ids[start:end]

        atom_mask = (atom_amino_id >= start) & (atom_amino_id < end)
        atom_chain_id = atom_chain_id[atom_mask]
        atom_amino_id = atom_amino_id[atom_mask] - start
        atom_names = atom_names[atom_mask]
        atom_types = atom_types[atom_mask]
        atom_pos = atom_pos[atom_mask]
        if atom_charge is not None:
            atom_charge = atom_charge[atom_mask]
        atom_radius = atom_radius[atom_mask]
        atom_ids = atom_ids[atom_mask]

        return (
            amino_types,
            atom_chain_id,
            atom_amino_id,
            atom_names,
            atom_types,
            atom_pos,
            atom_charge,
            atom_radius,
            res_sse,
            amino_ids,
            atom_ids,
        )

    @staticmethod
    def _strip_sidechains_to_ala(
        arrays: Tuple, positions, keep_cb: bool = True
    ) -> Tuple:
        """Replace sidechain atoms (beyond Cb) with nothing at the given
        residues — keep only N, CA, C, O, CB. Residue-level arrays unchanged.

        Used for S3F-style masked-residue leakage prevention: the surface and
        graph are built from coordinates where masked residues look like
        Alanine (backbone + Cb) when ``keep_cb`` is true. With ``keep_cb``
        false, all residues use the same N/CA/C/O template, avoiding the
        missing-CB glycine shortcut.
        """
        (
            amino_types,
            atom_chain_id,
            atom_amino_id,
            atom_names,
            atom_types,
            atom_pos,
            atom_charge,
            atom_radius,
            res_sse,
            amino_ids,
            atom_ids,
        ) = arrays

        positions_set = {int(p) for p in positions}
        retained_atom_names = BACKBONE_ATOM_NAMES
        if not keep_cb:
            retained_atom_names = BACKBONE_ATOM_NAMES - {"CB"}

        keep = np.ones(len(atom_amino_id), dtype=bool)
        for i in range(len(atom_amino_id)):
            res_id = int(atom_amino_id[i])
            if res_id in positions_set:
                name = str(atom_names[i]).strip().upper()
                if name not in retained_atom_names:
                    keep[i] = False

        atom_chain_id = atom_chain_id[keep]
        atom_amino_id = atom_amino_id[keep]
        atom_names = atom_names[keep]
        atom_types = atom_types[keep]
        atom_pos = atom_pos[keep]
        if atom_charge is not None:
            atom_charge = atom_charge[keep]
        atom_radius = atom_radius[keep]
        atom_ids = atom_ids[keep]

        return (
            amino_types,
            atom_chain_id,
            atom_amino_id,
            atom_names,
            atom_types,
            atom_pos,
            atom_charge,
            atom_radius,
            res_sse,
            amino_ids,
            atom_ids,
        )

    def _generate_surface(
        self,
        pdb_path: str,
        protein_name: str,
        pocket_name: Optional[str],
        parsed_arrays: Tuple,
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
        use_igl_normals = getattr(cfg, "use_igl_normals", False)
        nanoshaper_grid_scale = getattr(cfg, "nanoshaper_grid_scale", 0.3)
        edtsurf_grid_scale = getattr(cfg, "edtsurf_grid_scale", 0.5)
        use_poisson = getattr(cfg, "use_poisson", False)
        poisson_high_precision = getattr(cfg, "poisson_high_precision", True)
        tufting = getattr(cfg, "tufting", False)

        try:
            extra_kwargs = {}
            if (
                surface_method in ("alpha_complex", "nanoshaper")
                and parsed_arrays is not None
            ):
                extra_kwargs["atom_pos"] = parsed_arrays[5]
                extra_kwargs["atom_radius"] = parsed_arrays[7]

            uses_precomputed_msms = (
                surface_method == "msms" and precomputed_patches_dir is not None
            )
            should_extract_patch = not use_whole_surfaces and (
                pocket_name is not None or uses_precomputed_msms
            )

            if should_extract_patch:
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
                        pdb_to_edtsurf,
                        pdb_to_nanoshaper,
                        pdb_to_surf_with_min,
                    )

                    if surface_method == "msms":
                        verts, faces = pdb_to_surf_with_min(
                            pdb_path, min_number=min_vert_number
                        )
                    elif surface_method == "alpha_complex":
                        verts, faces = pdb_to_alpha_complex(
                            pdb_path,
                            alpha_value=alpha_value,
                            atom_pos=extra_kwargs.get("atom_pos"),
                            atom_radius=extra_kwargs.get("atom_radius"),
                        )
                    elif surface_method == "edtsurf":
                        verts, faces = pdb_to_edtsurf(
                            pdb_path, grid_scale=edtsurf_grid_scale
                        )
                    elif surface_method == "nanoshaper":
                        verts, faces = pdb_to_nanoshaper(
                            pdb_path,
                            grid_scale=nanoshaper_grid_scale,
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
                            logger.debug("Patch extraction failed for %s", pocket_name)
                            return None
                        patch_verts, patch_faces = result
                    else:
                        patch_verts, patch_faces = verts, faces

                # Apply mesh noise BEFORE computing operators (no-ops unless augmentor is independent)
                if self.noise_augmentor is not None:
                    patch_verts = self.noise_augmentor.apply_mesh_noise(
                        patch_verts, patch_faces
                    )

                surface = SurfaceObject.from_verts_faces(
                    verts=patch_verts,
                    faces=patch_faces,
                    face_reduction_rate=face_reduction_rate,
                    use_pymesh=use_pymesh,
                    surface_method=surface_method,
                    min_vert_number=min_vert_number,
                    use_igl_normals=use_igl_normals,
                    use_poisson=use_poisson,
                    poisson_high_precision=poisson_high_precision,
                    tufting=tufting,
                )

                surface.add_geom_feats()
            else:
                from alphasurf.protein.create_surface import (
                    pdb_to_alpha_complex,
                    pdb_to_edtsurf,
                    pdb_to_nanoshaper,
                    pdb_to_surf_with_min,
                )

                if surface_method == "msms":
                    verts, faces = pdb_to_surf_with_min(
                        pdb_path, min_number=min_vert_number
                    )
                elif surface_method == "alpha_complex":
                    verts, faces = pdb_to_alpha_complex(
                        pdb_path,
                        alpha_value=alpha_value,
                        atom_pos=extra_kwargs.get("atom_pos"),
                        atom_radius=extra_kwargs.get("atom_radius"),
                    )
                elif surface_method == "edtsurf":
                    verts, faces = pdb_to_edtsurf(
                        pdb_path, grid_scale=edtsurf_grid_scale
                    )
                elif surface_method == "nanoshaper":
                    verts, faces = pdb_to_nanoshaper(
                        pdb_path,
                        grid_scale=nanoshaper_grid_scale,
                        atom_pos=extra_kwargs.get("atom_pos"),
                        atom_radius=extra_kwargs.get("atom_radius"),
                    )
                else:
                    raise ValueError(f"Unknown surface method: {surface_method}")

                # Apply mesh noise BEFORE computing operators (no-ops unless augmentor is independent)
                if self.noise_augmentor is not None:
                    verts = self.noise_augmentor.apply_mesh_noise(verts, faces)

                surface = SurfaceObject.from_verts_faces(
                    verts=verts,
                    faces=faces,
                    face_reduction_rate=face_reduction_rate,
                    use_pymesh=use_pymesh,
                    surface_method=surface_method,
                    min_vert_number=min_vert_number,
                    use_igl_normals=use_igl_normals,
                    use_poisson=use_poisson,
                    poisson_high_precision=poisson_high_precision,
                    tufting=tufting,
                )

                surface.add_geom_feats()

            # Compute vertex-to-residue mapping + atom types
            if parsed_arrays is not None and surface is not None:
                atom_pos_np = parsed_arrays[5]
                atom_types_np = parsed_arrays[4]
                if atom_pos_np is not None and len(atom_pos_np) > 0:
                    verts_t = torch.from_numpy(surface.verts).float()
                    atom_pos_t = torch.from_numpy(atom_pos_np).float()
                    dists = torch.cdist(verts_t, atom_pos_t)
                    min_dists, closest_atoms = dists.min(dim=1)
                    exact_match = min_dists < 1e-6
                    vert_atom_ids = torch.full((len(verts_t),), -1, dtype=torch.long)
                    vert_atom_ids[exact_match] = closest_atoms[exact_match]
                    surface.vert_atom_ids = vert_atom_ids
                    vert_atom_types = torch.full((len(verts_t),), -1, dtype=torch.long)
                    vert_atom_types[exact_match] = torch.from_numpy(
                        atom_types_np[closest_atoms[exact_match].numpy()]
                    ).long()
                    surface.vert_atom_types = vert_atom_types

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
