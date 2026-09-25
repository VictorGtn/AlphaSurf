"""
Unified ProteinLoader supporting both disk and on-the-fly modes.

This module consolidates surface and graph loading with integrated transform
support. Transforms (noise, patch extraction) are applied DURING generation
so that computed features (operators, edges, etc.) reflect the transformed geometry.

Transform order: parse the PDB into raw arrays, apply atom noise (to both
branches under the joint modes, to the graph only under `independent`), generate
the mesh, extract the patch if configured, apply mesh noise (under
`independent`, `joint_mesh` and `alpha_joint_mesh`), compute operators on the
final mesh, build the graph from the same arrays, and expand features.
"""

import logging
import os
from typing import Any, Literal, Optional, Tuple

import numpy as np
import torch
from alphasurf.protein.graphs import (
    atom_type_dict,
    get_sbl_radius,
    parse_pdb_path,
    res_type_to_hphob,
)
from alphasurf.protein.protein import Protein
from alphasurf.protein.residue_graph import ResidueGraphBuilder
from alphasurf.protein.surfaces import SurfaceObject
from alphasurf.protein.transforms import NoiseAugmentor, PatchExtractor
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

BACKBONE_ATOM_NAMES = {"N", "CA", "C", "O", "CB"}

# Ideal alanine geometry from the Dunbrack-derived internal-coordinate
# statistics distributed with Biopython (Ala N-CA-CB and CB-CA-C hedrons).
# The placement itself follows the closed-form construction described in SBL's
# Molecular_coordinates manual, "Embedding the Cbeta carbon atom".
ALANINE_CA_CB_LENGTH = 1.52608
ALANINE_N_CA_CB_ANGLE_DEG = 110.40921
ALANINE_C_CA_CB_ANGLE_DEG = 110.28430


class ProteinLoader:
    """
    Load proteins either from disk or generate on-the-fly.

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
        self.read_b_factors = getattr(graph_config, "read_b_factors", True)

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
                atoms at these residue indices before surface/graph generation.
            ala_strip_keep_cb: when true, reconstruct the same idealized
                L-alanine C-beta from N/CA/C at every stripped position,
                including glycine. When false, use an N/CA/C/O-only mask.

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

        residue_b_factors = (
            self._read_residue_b_factors(pdb_path)
            if self.read_b_factors and not ala_strip_positions
            else None
        )
        parsed_arrays = self._parse_pdb(pdb_path)
        if parsed_arrays is None:
            return None

        if residue_b_factors is not None and len(residue_b_factors) != len(
            parsed_arrays[0]
        ):
            logger.warning(
                "PDB residue/B-factor count mismatch for %s: %d vs %d",
                pdb_path,
                len(residue_b_factors),
                len(parsed_arrays[0]),
            )
            residue_b_factors = None

        if crop_window is not None:
            parsed_arrays = self._crop_parsed_arrays(parsed_arrays, *crop_window)
            if residue_b_factors is not None:
                residue_b_factors = residue_b_factors[crop_window[0] : crop_window[1]]

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

        from alphasurf.utils.timing_stats import Timer

        with Timer("surface_pipeline"):
            surface = self._generate_surface(
                pdb_path=pdb_path,
                protein_name=protein_name,
                pocket_name=pocket_name,
                parsed_arrays=parsed_for_surface,
                alpha_override=alpha_override,
            )

        with Timer("graph_pipeline"):
            graph = self._generate_graph(
                pdb_path=pdb_path,
                protein_name=protein_name,
                parsed_arrays=parsed_for_graph,
            )

        if (
            graph is not None
            and residue_b_factors is not None
            and hasattr(graph, "node_pos")
        ):
            if len(residue_b_factors) == len(graph.node_pos):
                graph.b_factor = torch.from_numpy(residue_b_factors).float()
            else:
                logger.warning(
                    "Generated graph/B-factor count mismatch for %s: %d vs %d",
                    pdb_path,
                    len(residue_b_factors),
                    len(graph.node_pos),
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
    def _read_residue_b_factors(pdb_path: str) -> Optional[np.ndarray]:
        """Read one representative B-factor for each standard PDB residue."""
        from Bio.PDB import PDBParser

        try:
            structure = PDBParser(QUIET=True).get_structure("plddt", pdb_path)
        except Exception as error:
            logger.warning("Could not read PDB B-factors from %s: %s", pdb_path, error)
            return None

        values = []
        for residue in structure.get_residues():
            if residue.id[0] != " ":
                continue
            atoms = list(residue.get_atoms())
            ca = next((atom for atom in atoms if atom.get_name() == "CA"), None)
            if ca is not None:
                values.append(float(ca.get_bfactor()))
                continue
            atom_values = [float(atom.get_bfactor()) for atom in atoms]
            values.append(float(np.mean(atom_values)) if atom_values else np.nan)

        if not values:
            return None
        return np.asarray(values, dtype=np.float32)

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
    def _embed_alanine_cb(
        n_pos: np.ndarray,
        ca_pos: np.ndarray,
        c_pos: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Place an ideal L-alanine C-beta from N, CA and C coordinates.

        Let ``u`` and ``v`` be the unit CA->N and CA->C vectors and write the
        unit CA->CB vector as ``w = a*u + b*v + g*n``, where
        ``n = normalize(u x v)``. Imposing the two SBL valence-angle
        constraints gives::

            a = (cos(theta_N) - cos(theta_C) cos(theta)) / sin(theta)^2
            b = (cos(theta_C) - cos(theta_N) cos(theta)) / sin(theta)^2
            g = sqrt(1 - a^2 - b^2 - 2ab cos(theta))

        The positive root selects the canonical L-amino-acid chirality because
        ``(N, C, CA, CB)`` then has positive oriented tetrahedral volume.
        """
        n_pos = np.asarray(n_pos, dtype=np.float64)
        ca_pos = np.asarray(ca_pos, dtype=np.float64)
        c_pos = np.asarray(c_pos, dtype=np.float64)
        if not (
            np.isfinite(n_pos).all()
            and np.isfinite(ca_pos).all()
            and np.isfinite(c_pos).all()
        ):
            return None

        u = n_pos - ca_pos
        v = c_pos - ca_pos
        u_norm = float(np.linalg.norm(u))
        v_norm = float(np.linalg.norm(v))
        if u_norm < 1e-8 or v_norm < 1e-8:
            return None
        u /= u_norm
        v /= v_norm

        cos_theta = float(np.clip(np.dot(u, v), -1.0, 1.0))
        sin_theta_sq = 1.0 - cos_theta * cos_theta
        normal = np.cross(u, v)
        normal_norm = float(np.linalg.norm(normal))
        if sin_theta_sq < 1e-8 or normal_norm < 1e-8:
            return None
        normal /= normal_norm

        cos_theta_n = float(np.cos(np.deg2rad(ALANINE_N_CA_CB_ANGLE_DEG)))
        cos_theta_c = float(np.cos(np.deg2rad(ALANINE_C_CA_CB_ANGLE_DEG)))
        coeff_u = (cos_theta_n - cos_theta_c * cos_theta) / sin_theta_sq
        coeff_v = (cos_theta_c - cos_theta_n * cos_theta) / sin_theta_sq
        coeff_normal_sq = (
            1.0
            - coeff_u * coeff_u
            - coeff_v * coeff_v
            - 2.0 * coeff_u * coeff_v * cos_theta
        )
        if coeff_normal_sq < -1e-6:
            return None
        coeff_normal = float(np.sqrt(max(0.0, coeff_normal_sq)))

        ca_to_cb = coeff_u * u + coeff_v * v + coeff_normal * normal
        cb_pos = ca_pos + ALANINE_CA_CB_LENGTH * ca_to_cb
        return cb_pos.astype(np.float32)

    @staticmethod
    def _strip_sidechains_to_ala(
        arrays: Tuple, positions, keep_cb: bool = True
    ) -> Tuple:
        """Standardize masked residues to alanine or backbone-only geometry.

        All native sidechain atoms, including the native C-beta, are removed.
        If ``keep_cb`` is true, a new idealized L-alanine C-beta is synthesized
        from N/CA/C for every selected residue, including glycine. Thus neither
        native C-beta deviations nor C-beta absence can leak residue identity.
        If ``keep_cb`` is false, only N/CA/C/O are retained.
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
        retained_atom_names = BACKBONE_ATOM_NAMES - {"CB"}

        pseudo_cb_records = []
        if keep_cb:
            for res_id in sorted(positions_set):
                residue_indices = np.flatnonzero(atom_amino_id == res_id)
                coords = {
                    str(atom_names[i]).strip().upper(): atom_pos[i]
                    for i in residue_indices
                }
                if not all(name in coords for name in ("N", "CA", "C")):
                    logger.warning(
                        "Cannot build alanine C-beta for residue %d: "
                        "missing N, CA, or C",
                        res_id,
                    )
                    continue
                cb_pos = ProteinLoader._embed_alanine_cb(
                    coords["N"], coords["CA"], coords["C"]
                )
                if cb_pos is None:
                    logger.warning(
                        "Cannot build alanine C-beta for residue %d: "
                        "degenerate backbone geometry",
                        res_id,
                    )
                    continue
                ca_matches = [
                    i
                    for i in residue_indices
                    if str(atom_names[i]).strip().upper() == "CA"
                ]
                if not ca_matches:
                    continue
                ca_index = ca_matches[0]
                pseudo_cb_records.append(
                    (
                        atom_chain_id[ca_index],
                        res_id,
                        cb_pos,
                        f"{amino_ids[res_id]}_CB",
                    )
                )

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

        if pseudo_cb_records:
            n_new = len(pseudo_cb_records)
            atom_chain_id = np.concatenate(
                [
                    atom_chain_id,
                    np.asarray(
                        [record[0] for record in pseudo_cb_records],
                        dtype=atom_chain_id.dtype,
                    ),
                ]
            )
            atom_amino_id = np.concatenate(
                [
                    atom_amino_id,
                    np.asarray(
                        [record[1] for record in pseudo_cb_records],
                        dtype=atom_amino_id.dtype,
                    ),
                ]
            )
            atom_names = np.concatenate(
                [atom_names, np.full(n_new, "CB", dtype=atom_names.dtype)]
            )
            atom_types = np.concatenate(
                [
                    atom_types,
                    np.full(
                        n_new,
                        atom_type_dict["C"],
                        dtype=atom_types.dtype,
                    ),
                ]
            )
            atom_pos = np.concatenate(
                [
                    atom_pos,
                    np.stack([record[2] for record in pseudo_cb_records]).astype(
                        atom_pos.dtype, copy=False
                    ),
                ],
                axis=0,
            )
            if atom_charge is not None:
                atom_charge = np.concatenate(
                    [
                        atom_charge,
                        np.zeros(n_new, dtype=atom_charge.dtype),
                    ]
                )
            atom_radius = np.concatenate(
                [
                    atom_radius,
                    np.full(
                        n_new,
                        get_sbl_radius("CB", "ALA", "C"),
                        dtype=atom_radius.dtype,
                    ),
                ]
            )
            atom_ids = np.concatenate(
                [
                    atom_ids,
                    np.asarray(
                        [record[3] for record in pseudo_cb_records],
                        dtype=atom_ids.dtype,
                    ),
                ]
            )

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
        edtsurf_surface_mode = getattr(cfg, "edtsurf_surface_mode", 2)
        use_poisson = getattr(cfg, "use_poisson", False)
        poisson_high_precision = getattr(cfg, "poisson_high_precision", True)
        tufting = getattr(cfg, "tufting", False)

        try:
            if surface_method == "patch_graph":
                return self._generate_patch_graph_surface(
                    cfg=cfg,
                    protein_name=protein_name,
                    parsed_arrays=parsed_arrays,
                    pocket_name=pocket_name,
                    alpha_value=alpha_value,
                )

            extra_kwargs = {}
            if (
                surface_method in ("alpha_complex", "nanoshaper", "msms")
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

                    from alphasurf.utils.timing_stats import Timer

                    with Timer("surface_mesh_generation"):
                        if surface_method == "msms":
                            verts, faces = pdb_to_surf_with_min(
                                pdb_path,
                                min_number=min_vert_number,
                                atom_pos=extra_kwargs.get("atom_pos"),
                                atom_radius=extra_kwargs.get("atom_radius"),
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
                                pdb_path,
                                grid_scale=edtsurf_grid_scale,
                                surface_mode=edtsurf_surface_mode,
                            )
                        elif surface_method == "nanoshaper":
                            verts, faces = pdb_to_nanoshaper(
                                pdb_path,
                                grid_scale=nanoshaper_grid_scale,
                                atom_pos=extra_kwargs.get("atom_pos"),
                                atom_radius=extra_kwargs.get("atom_radius"),
                            )
                        else:
                            raise ValueError(
                                f"Unknown surface method: {surface_method}"
                            )

                    if self.patch_extractor is not None and pocket_name is not None:
                        with Timer("patch_extraction"):
                            result = self.patch_extractor.extract_patch(
                                verts, faces, pocket_name
                            )
                        if result is None:
                            logger.debug("Patch extraction failed for %s", pocket_name)
                            return None
                        patch_verts, patch_faces = result
                    else:
                        patch_verts, patch_faces = verts, faces

                # Mesh noise must land before the operators are computed. A no-op
                # unless the mode is independent, joint_mesh or alpha_joint_mesh.
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
                        pdb_path,
                        min_number=min_vert_number,
                        atom_pos=extra_kwargs.get("atom_pos"),
                        atom_radius=extra_kwargs.get("atom_radius"),
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
                        pdb_path,
                        grid_scale=edtsurf_grid_scale,
                        surface_mode=edtsurf_surface_mode,
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

                # Mesh noise must land before the operators are computed. A no-op
                # unless the mode is independent, joint_mesh or alpha_joint_mesh.
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

            # Map each vertex to the atom it coincides with, if any. Alpha-complex
            # vertices sit exactly on atom centres; other methods leave this -1.
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

    def _generate_patch_graph_surface(
        self,
        cfg,
        protein_name: str,
        parsed_arrays: Tuple,
        pocket_name: Optional[str],
        alpha_value: float,
    ) -> Optional[SurfaceObject]:
        """Build a DiffusionNet-compatible surface from SBL spherical patches."""
        if parsed_arrays is None:
            return None

        from alphasurf.protein.patch_operators import (
            build_patch_operators,
            extract_patch_graph,
            induced_patch_subgraph,
            load_patch_graph,
        )

        amino_types = parsed_arrays[0]
        atom_amino_id = parsed_arrays[2]
        atom_types = parsed_arrays[4]
        atom_pos = parsed_arrays[5]
        atom_charge = parsed_arrays[6]
        atom_radius = parsed_arrays[7]
        probe_radius = getattr(cfg, "patch_probe_radius", 1.4)
        k_eig = getattr(cfg, "k_eig", 128)

        patch_graph_dir = getattr(cfg, "patch_graph_dir", None)
        cache_path = (
            os.path.join(str(patch_graph_dir), f"{protein_name}.npz")
            if patch_graph_dir
            else None
        )
        cache_is_compatible = not (
            self.noise_augmentor is not None and self.noise_augmentor.enabled
        )
        if cache_path and cache_is_compatible and os.path.exists(cache_path):
            patch_graph = load_patch_graph(
                cache_path,
                alpha=alpha_value,
                probe_radius=probe_radius,
                atom_positions=atom_pos,
                atom_radii=atom_radius,
            )
        else:
            patch_graph = extract_patch_graph(
                atom_pos,
                atom_radius,
                alpha=alpha_value,
                probe_radius=probe_radius,
            )

        use_whole_surfaces = getattr(cfg, "use_whole_surfaces", True)
        if (
            not use_whole_surfaces
            and pocket_name is not None
            and self.patch_extractor is not None
        ):
            reference_vertices = self.patch_extractor.get_patch_vertices(pocket_name)
            if reference_vertices is None:
                return None

            from scipy.spatial import cKDTree

            distances, _ = cKDTree(reference_vertices).query(
                patch_graph.patch_center, k=1
            )
            min_patches = getattr(cfg, "patch_graph_min_patches", 16)
            radius = self.patch_extractor.radius
            # A null max_radius means unbounded, as in PatchExtractor.extract_patch.
            max_radius = self.patch_extractor.max_radius
            if max_radius is None:
                max_radius = float(distances.max())
            selected_mask = None
            while radius <= max_radius:
                candidate_mask = distances <= radius
                if np.count_nonzero(candidate_mask) >= min_patches:
                    selected_mask = self._largest_patch_graph_component(
                        patch_graph, candidate_mask
                    )
                    if np.count_nonzero(selected_mask) >= min_patches:
                        break
                radius += 2.0

            if selected_mask is None or np.count_nonzero(selected_mask) < min_patches:
                logger.warning(
                    "Patch-graph extraction failed for %s: fewer than %d patches "
                    "within %.1f A",
                    pocket_name,
                    min_patches,
                    max_radius,
                )
                return None
            patch_graph = induced_patch_subgraph(patch_graph, selected_mask)

        operators = build_patch_operators(patch_graph, k_eig=k_eig)
        fields = operators.diffusionnet_fields()
        n_patches = patch_graph.num_patches

        # Curvature is constant per atom type on spherical patches, so the
        # per-patch shape is described by its extent instead.
        area = patch_graph.patch_area
        exposed_fraction = area / (4.0 * np.pi * patch_graph.patch_radius**2)
        boundary_length = np.zeros(n_patches)
        np.add.at(boundary_length, patch_graph.edge_index[0], patch_graph.shared_arc_length)
        np.add.at(boundary_length, patch_graph.edge_index[1], patch_graph.shared_arc_length)
        # Areas and boundary ratios span several orders of magnitude.
        log_area = np.log(area)
        log_boundary_area_ratio = np.log(boundary_length / area)

        # Spherical patches are convex inside; concavity lives on the arcs where
        # the normal jumps. Signed dihedral angle per arc (negative = concave),
        # averaged over a patch's arcs weighted by arc length.
        source, target = patch_graph.edge_index
        normals = patch_graph.patch_normal
        centers = patch_graph.patch_center
        cos_angle = np.clip(np.sum(normals[source] * normals[target], axis=1), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        orientation = np.sum(
            (normals[target] - normals[source]) * (centers[target] - centers[source]), axis=1
        )
        angle = np.where(orientation < 0.0, -angle, angle)
        both_valid = patch_graph.patch_normal_valid[source] & patch_graph.patch_normal_valid[target]
        weighted_angle = np.where(both_valid, angle * patch_graph.shared_arc_length, 0.0)
        junction_angle = np.zeros(n_patches)
        np.add.at(junction_angle, source, weighted_angle)
        np.add.at(junction_angle, target, weighted_angle)
        junction_angle /= np.maximum(boundary_length, 1e-12)

        hks_times = np.geomspace(0.1, 1000.0, 16)
        hks_phase = np.exp(-operators.eigenvalues[None, :] * hks_times[:, None])
        hks = (operators.eigenvectors**2) @ hks_phase.T
        hks /= np.maximum(hks.mean(axis=0, keepdims=True), 1e-12)

        geometry_features = np.concatenate(
            (
                log_area[:, None],
                exposed_fraction[:, None],
                log_boundary_area_ratio[:, None],
                junction_angle[:, None],
                hks,
                normals,
            ),
            axis=1,
        ).astype(np.float32)

        surface = SurfaceObject(
            verts=patch_graph.patch_center.astype(np.float32),
            faces=np.empty((0, 3), dtype=np.int64),
            mass=fields["mass"],
            L=fields["L"],
            evals=fields["evals"],
            evecs=fields["evecs"],
            gradX=fields["gradX"],
            gradY=fields["gradY"],
            vnormals=normals.astype(np.float32),
        )
        surface.features.add_named_features("geom_feats", geometry_features)

        # Each patch belongs to one atom, so atom chemistry is exact on the node.
        parent = patch_graph.patch_atom_index
        surface.features.add_named_oh_features(
            "atom_types", np.asarray(atom_types)[parent], nclasses=12
        )
        hphob = np.asarray(
            [res_type_to_hphob[amino_types[atom_amino_id[i]]] for i in parent],
            dtype=np.float32,
        )
        surface.features.add_named_features("hphobs", hphob[:, None])
        if atom_charge is not None:
            surface.features.add_named_features(
                "charge", np.asarray(atom_charge, dtype=np.float32)[parent].reshape(-1, 1)
            )
        surface.drop_ratio = 0.0
        surface.drop_ratio_vertex = 0.0
        surface.from_numpy()

        parent_atoms = torch.from_numpy(patch_graph.patch_atom_index).long()
        surface.vert_atom_ids = parent_atoms
        surface.vert_atom_types = torch.from_numpy(
            np.asarray(atom_types)[patch_graph.patch_atom_index]
        ).long()
        surface.patch_area = torch.from_numpy(patch_graph.patch_area.astype(np.float32))
        surface.patch_radius = torch.from_numpy(
            patch_graph.patch_radius.astype(np.float32)
        )
        surface.patch_sphere_center = torch.from_numpy(
            patch_graph.patch_sphere_center.astype(np.float32)
        )
        surface.patch_area_centroid = torch.from_numpy(
            patch_graph.patch_area_centroid.astype(np.float32)
        )
        surface.patch_normal_valid = torch.from_numpy(patch_graph.patch_normal_valid)
        surface.patch_arc_count = torch.from_numpy(
            patch_graph.arc_count.astype(np.int64)
            if patch_graph.arc_count is not None
            else np.ones(patch_graph.num_edges, dtype=np.int64)
        )
        surface.patch_edge_index = torch.from_numpy(
            patch_graph.edge_index.astype(np.int64)
        )
        surface.shared_arc_length = torch.from_numpy(
            patch_graph.shared_arc_length.astype(np.float32)
        )

        with torch.no_grad():
            surface.expand_features(
                remove_feats=True,
                feature_keys=self._surface_feat_keys,
                oh_keys=self._surface_oh_keys,
            )
        return surface

    @staticmethod
    def _largest_patch_graph_component(
        patch_graph, node_mask: np.ndarray
    ) -> np.ndarray:
        """Keep the largest connected component of a masked patch graph."""
        import scipy.sparse

        selected = np.flatnonzero(node_mask)
        if selected.size == 0:
            return node_mask

        old_to_new = np.full(patch_graph.num_patches, -1, dtype=np.int64)
        old_to_new[selected] = np.arange(selected.size)
        source, target = patch_graph.edge_index
        edge_mask = node_mask[source] & node_mask[target]
        local_source = old_to_new[source[edge_mask]]
        local_target = old_to_new[target[edge_mask]]
        adjacency = scipy.sparse.coo_matrix(
            (
                np.ones(2 * local_source.size),
                (
                    np.concatenate((local_source, local_target)),
                    np.concatenate((local_target, local_source)),
                ),
            ),
            shape=(selected.size, selected.size),
        )
        _, labels = scipy.sparse.csgraph.connected_components(adjacency, directed=False)
        component_areas = np.bincount(labels, weights=patch_graph.patch_area[selected])
        keep_label = int(np.argmax(component_areas))
        output_mask = np.zeros(patch_graph.num_patches, dtype=bool)
        output_mask[selected[labels == keep_label]] = True
        return output_mask

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
