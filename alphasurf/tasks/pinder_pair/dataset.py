"""
PINDER-Pair dataset for protein-protein interaction prediction.

Uses the PINDER dataset to load protein pairs and predict interface residues.
"""

import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import biotite.sequence as seq
import biotite.sequence.align as align
import numpy as np
import torch
from alphasurf.protein.protein_loader import ProteinLoader
from Bio import SeqUtils
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _generate_negative_pairs(
    pos_pairs: np.ndarray, n_res1: int, n_res2: int
) -> np.ndarray:
    """
    Generate all (i, j) residue pairs NOT in pos_pairs, vectorized.

    Args:
        pos_pairs: (2, K) array of positive pair indices.
        n_res1: Number of residues in protein 1.
        n_res2: Number of residues in protein 2.

    Returns:
        (2, M) array of negative pair indices.
    """
    all_pairs = np.mgrid[0:n_res1, 0:n_res2].reshape(2, -1)
    pos_flat = pos_pairs[0] * n_res2 + pos_pairs[1]
    all_flat = all_pairs[0] * n_res2 + all_pairs[1]
    neg_mask = ~np.isin(all_flat, pos_flat)
    return all_pairs[:, neg_mask]


def load_pinder_split(
    data_dir: str,
    split: str = "train",
    test_setting: Optional[str] = None,
    max_systems: Optional[int] = None,
) -> List[Dict]:
    """
    Load PINDER dataset split.

    Looks for data in this order:
    1. systems_{split}_{setting}.csv (for test with specific setting)
    2. systems_{split}.csv in data_dir
    3. index.parquet in data_dir (filtered by split)
    4. pinder package (if installed)

    Args:
        data_dir: Directory containing pinder data
        split: One of 'train', 'val', 'test'
        test_setting: For test split, one of 'holo', 'apo', 'af2'
        max_systems: Limit number of systems to load (for debugging)

    Returns:
        List of dicts with keys: id, receptor_id, ligand_id
    """
    import pandas as pd

    data_dir = Path(data_dir) if not isinstance(data_dir, Path) else data_dir

    # For test split, try setting-specific file first
    if split == "test" and test_setting:
        csv_path = data_dir / f"systems_{split}_{test_setting}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if max_systems is not None:
                df = df.iloc[:max_systems]
            return _df_to_systems(df)

    # Try split-specific CSV
    csv_path = data_dir / f"systems_{split}.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if max_systems is not None:
            df = df.iloc[:max_systems]
        return _df_to_systems(df)

    # Try index.parquet
    parquet_path = data_dir / "index.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "split" in df.columns:
            df = df[df["split"] == split]
        return _df_to_systems(df)

    # Try pinder package
    try:
        from pinder.core import get_index
        from pinder.core.loader.loader import PinderLoader

        print(f"Loading PINDER split '{split}' using pinder package...")

        # Get index and filter by split
        index = get_index()
        split_df = index[index["split"] == split]

        # Cluster sampling: Pick one representative at random per cluster
        # Using random_state=42 to ensure stability (doesn't change epoch from epoch)
        print(f"Sampling 1 representative per cluster from {len(split_df)} systems...")

        # Shuffle deterministically, then take the first of each cluster
        # This is equivalent to random sampling per group but more robust/faster
        shuffled_df = split_df.sample(frac=1, random_state=42)
        sampled_df = shuffled_df.drop_duplicates(subset=["cluster_id"])

        selected_ids = sampled_df["id"].tolist()
        print(f"Found {len(selected_ids)} clusters.")

        if max_systems is not None:
            print(
                f"Limiting to first {max_systems} systems for debugging/verification."
            )
            selected_ids = selected_ids[:max_systems]

        print(f"Selected {len(selected_ids)} systems.")

        loader = PinderLoader(ids=selected_ids)
        systems = []

        for item in loader:
            # PinderLoader might yield (PinderSystem, Structure, Structure) tuple
            if isinstance(item, tuple):
                system = item[0]
            else:
                system = item

            # Determine paths based on split/setting
            r_path, l_path = _resolve_pinder_paths(system, split, test_setting)

            if r_path and l_path and os.path.exists(r_path) and os.path.exists(l_path):
                systems.append(
                    {
                        "id": system.entry.id,
                        "receptor_id": f"{system.entry.id}_R",
                        "ligand_id": f"{system.entry.id}_L",
                        "receptor_path": str(r_path),
                        "ligand_path": str(l_path),
                    }
                )
        print(f"Loaded {len(systems)} systems from pinder package")
        return systems
    except ImportError as e:
        print(f"FAILED to import PinderLoader: {e}")
        pass
    except Exception as e:
        print(f"FAILED during PinderLoader iteration: {e}")
        import traceback

        traceback.print_exc()
        pass

    raise FileNotFoundError(
        f"No PINDER data found in {data_dir}. "
        f"Run preprocess.py first or install pinder package."
    )


def _resolve_pinder_paths(
    system, split: str, setting: str = None
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve receptor and ligand paths from PinderSystem."""
    # Default to holo for train/val
    if split != "test":
        setting = "holo"
    if setting is None:
        setting = "holo"

    r_path = None
    l_path = None

    if setting == "holo":
        if system.holo_receptor:
            r_path = system.holo_receptor.filepath
        if system.holo_ligand:
            l_path = system.holo_ligand.filepath
    elif setting == "apo":
        if system.apo_receptor:
            r_path = system.apo_receptor.filepath
        if system.apo_ligand:
            l_path = system.apo_ligand.filepath
    elif setting == "af2":
        if system.pred_receptor:
            r_path = system.pred_receptor.filepath
        if system.pred_ligand:
            l_path = system.pred_ligand.filepath

    # Fallback for test: if specific setting missing, fall back to holo?
    # Usually strictly enforcing the setting is better for evaluation correctness.
    # But let's check if we want fallback. The preprocess.py had fallback.
    # checking preprocess.py... yes, it falls back to holo.
    if split == "test" and (r_path is None or l_path is None):
        if system.holo_receptor:
            r_path = system.holo_receptor.filepath
        if system.holo_ligand:
            l_path = system.holo_ligand.filepath

    return r_path, l_path


def _df_to_systems(df) -> List[Dict]:
    """Convert DataFrame to list of system dicts."""
    systems = []
    for _, row in df.iterrows():
        system_id = row.get("id", str(row.name))
        systems.append(
            {
                "id": system_id,
                "receptor_id": row.get("receptor_id", f"{system_id}_R"),
                "ligand_id": row.get("ligand_id", f"{system_id}_L"),
                "setting": row.get("setting"),
                # Allow pre-computed paths if in CSV
                "receptor_path": row.get("receptor_path"),
                "ligand_path": row.get("ligand_path"),
            }
        )
    return systems


class PinderPairDataset(Dataset):
    """
    Dataset for PINDER protein-protein interaction prediction.

    Each sample contains two proteins (receptor and ligand) with:
    - Surface meshes for both
    - Residue graphs for both
    - Interface labels (which residue pairs interact)

    Similar to PIP but uses PINDER dataset format.
    """

    def __init__(
        self,
        systems: List[Dict],
        protein_loader: ProteinLoader,
        pdb_dir: str,
        apply_noise: bool = False,
        neg_to_pos_ratio: float = 1.0,
        max_pos_per_pair: int = -1,
        interface_distance_graph: float = 8.0,
        interface_distance_surface: float = 8.0,
        # Legacy/Compatibility argument
        interface_distance: float = 8.0,
        surface_neg_to_pos_ratio: float = 10.0,
    ):
        """
        Args:
            systems: List of system dicts from load_pinder_split()
            protein_loader: ProteinLoader instance
            pdb_dir: Directory containing PDB files
            apply_noise: Whether to apply noise augmentation
            neg_to_pos_ratio: Ratio of negative to positive residue pairs to sample
            max_pos_per_pair: Maximum positive pairs per system (-1 for all)
            interface_distance_graph: Distance threshold (Å) for residue graph interface
            interface_distance_surface: Distance threshold (Å) for surface mesh interface
        """
        self.systems = systems
        self.protein_loader = protein_loader
        self.pdb_dir = pdb_dir
        self.apply_noise = apply_noise
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_per_pair = max_pos_per_pair

        self.interface_distance_graph = interface_distance_graph
        self.interface_distance_surface = interface_distance_surface

        self.surface_neg_to_pos_ratio = surface_neg_to_pos_ratio

    def __len__(self) -> int:
        return len(self.systems)

    def _get_pdb_path(self, system: Dict, protein_type: str) -> str:
        """Get PDB path for receptor or ligand."""
        # 1. Check for explicit path
        path_key = f"{protein_type}_path"
        if path_key in system and system[path_key]:
            return system[path_key]

        # 2. Check for explicit pdb key (legacy)
        if f"{protein_type}_pdb" in system:
            return system[f"{protein_type}_pdb"]

        # 3. Construct from ID and directory
        system_id = system["id"]
        protein_id = system.get(f"{protein_type}_id", system_id)
        setting = system.get("setting")

        # Check common naming patterns in pdb_dir
        candidates = []

        # Try suffixed names first (e.g. _R_apo.pdb)
        if setting:
            # protein_id is usually {id}_R, so we try {id}_R_{setting}.pdb
            candidates.append(os.path.join(self.pdb_dir, f"{protein_id}_{setting}.pdb"))
            # or if protein_id is just the uniprot, and we have system_id
            candidates.append(
                os.path.join(self.pdb_dir, f"{system_id}_{protein_type}_{setting}.pdb")
            )

        # Standard names (e.g. _R.pdb)
        candidates.extend(
            [
                os.path.join(self.pdb_dir, f"{system_id}_{protein_type}.pdb"),
                os.path.join(self.pdb_dir, f"{protein_id}.pdb"),
                os.path.join(self.pdb_dir, protein_type, f"{system_id}.pdb"),
            ]
        )

        for path in candidates:
            if os.path.exists(path):
                return path

        return candidates[0]  # Return first candidate even if doesn't exist

    def _count_graph_components(self, graph):
        """Count connected components in a graph using BFS."""
        n_nodes = len(graph.node_pos)
        visited = torch.zeros(n_nodes, dtype=torch.bool)
        n_components = 0

        # Build adjacency list from edge_index
        adj_list = [[] for _ in range(n_nodes)]
        if hasattr(graph, "edge_index") and graph.edge_index is not None:
            edges = graph.edge_index.t().tolist()
            for i, j in edges:
                adj_list[i].append(j)
                adj_list[j].append(i)

        # BFS to find components
        for start in range(n_nodes):
            if visited[start]:
                continue
            n_components += 1
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                for neighbor in adj_list[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)

        return n_components

    def _compute_interface(
        self,
        pos_1: torch.Tensor,
        pos_2: torch.Tensor,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute interface residue pairs based on distance.

        Args:
            pos_1: (N, 3) positions of protein 1 residues (CA atoms)
            pos_2: (M, 3) positions of protein 2 residues (CA atoms)
            threshold: Distance threshold in Angstroms

        Returns:
            Tuple of (positive_pairs, negative_pairs) each shape (2, K)
        """
        # Compute pairwise distances
        dists = torch.cdist(pos_1, pos_2)

        # Interface: residue pairs within threshold
        interface_mask = dists < threshold
        pos_i, pos_j = torch.where(interface_mask)
        pos_pairs = torch.stack([pos_i, pos_j]).numpy()

        # Non-interface pairs
        neg_i, neg_j = torch.where(~interface_mask)
        neg_pairs = torch.stack([neg_i, neg_j]).numpy()

        return pos_pairs, neg_pairs

    def _compute_interface_full_atom(
        self,
        atom_pos_1: torch.Tensor,
        atom_pos_2: torch.Tensor,
        res_map_1: torch.Tensor,
        res_map_2: torch.Tensor,
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute interface residue pairs based on ANY atom-atom distance.

        Args:
            atom_pos_1: (N_atoms1, 3) all atom positions of protein 1
            atom_pos_2: (N_atoms2, 3) all atom positions of protein 2
            res_map_1: (N_atoms1,) residue index for each atom in protein 1
            res_map_2: (N_atoms2,) residue index for each atom in protein 2
            threshold: Distance threshold in Angstroms

        Returns:
            Tuple of (positive_pairs, negative_pairs) each shape (2, K)
        """
        # Compute pairwise atom-atom distances
        atom_dists = torch.cdist(atom_pos_1, atom_pos_2)  # (N_atoms1, N_atoms2)

        # Find atom pairs within threshold
        close_atoms = atom_dists < threshold
        atom_i, atom_j = torch.where(close_atoms)

        # Map atoms to residues
        res_i = res_map_1[atom_i]
        res_j = res_map_2[atom_j]

        # Get unique residue pairs (interface)
        res_pairs = torch.stack([res_i, res_j], dim=1)
        unique_res_pairs = torch.unique(res_pairs, dim=0)

        pos_pairs = unique_res_pairs.T.numpy()  # (2, K)

        n_res1 = int(res_map_1.max() + 1)
        n_res2 = int(res_map_2.max() + 1)
        neg_pairs = _generate_negative_pairs(pos_pairs, n_res1, n_res2)

        return pos_pairs, neg_pairs

    def _sample_pairs(
        self,
        pos_pairs: np.ndarray,
        neg_pairs: np.ndarray,
        neg_to_pos_ratio: float = None,
        max_pos: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample balanced positive and negative pairs.

        Returns:
            (idx_left, idx_right, labels) tensors
        """
        ratio = (
            neg_to_pos_ratio if neg_to_pos_ratio is not None else self.neg_to_pos_ratio
        )
        max_p = max_pos if max_pos is not None else self.max_pos_per_pair

        num_pos = pos_pairs.shape[1]
        num_neg = neg_pairs.shape[1]

        if num_pos == 0:
            return None, None, None

        # Determine number of samples
        if ratio == -1:
            num_pos_use = num_pos
            num_neg_use = num_neg
        else:
            num_pos_use = min(num_pos, int(num_neg / ratio))
            if max_p > 0:
                num_pos_use = min(num_pos_use, max_p)
            num_neg_use = int(num_pos_use * ratio)

        num_pos_use = max(1, int(math.ceil(num_pos_use)))
        num_neg_use = max(1, int(math.ceil(num_neg_use)))

        # Sample
        pos_idx = np.random.choice(pos_pairs.shape[1], size=num_pos_use, replace=False)
        neg_idx = np.random.choice(
            neg_pairs.shape[1], size=min(num_neg_use, num_neg), replace=False
        )

        pos_sampled = pos_pairs[:, pos_idx]
        neg_sampled = neg_pairs[:, neg_idx]

        idx_left = torch.from_numpy(np.concatenate([pos_sampled[0], neg_sampled[0]]))
        idx_right = torch.from_numpy(np.concatenate([pos_sampled[1], neg_sampled[1]]))
        labels = torch.cat(
            [
                torch.ones(len(pos_sampled[0])),
                torch.zeros(len(neg_sampled[0])),
            ]
        )

        return idx_left, idx_right, labels

    def __getitem__(self, idx: int) -> Optional[Data]:
        system = self.systems[idx]

        # Load both proteins
        receptor_name = system.get("receptor_id", f"{system['id']}_R")
        ligand_name = system.get("ligand_id", f"{system['id']}_L")

        # Resolve paths (either from system dict or default logic)
        receptor_path = self._get_pdb_path(system, "receptor")
        ligand_path = self._get_pdb_path(system, "ligand")

        protein_1 = self.protein_loader.load(
            receptor_name, pdb_path=receptor_path, apply_noise=self.apply_noise
        )
        protein_2 = self.protein_loader.load(
            ligand_name, pdb_path=ligand_path, apply_noise=self.apply_noise
        )

        if protein_1 is None or protein_2 is None:
            # print(
            #     f"DEBUG: protein_1 or protein_2 is None for {receptor_name}/{ligand_name}"
            # )
            return None

        # Validate sizes
        if not protein_1.has_graph() or not protein_2.has_graph():
            # print(f"DEBUG: missing graph for {receptor_name}/{ligand_name}")
            return None

        g1_len = len(protein_1.graph.node_pos)
        g2_len = len(protein_2.graph.node_pos)

        if g1_len < 20 or g2_len < 20:
            # print(f"DEBUG: proteins too small: {g1_len}/{g2_len}")
            return None

        # Compute interface using full atom distances
        # Check if atom-level data is available in metadata
        has_atom_data_1 = (
            "atom_pos" in protein_1.metadata and "atom_res_map" in protein_1.metadata
        )
        has_atom_data_2 = (
            "atom_pos" in protein_2.metadata and "atom_res_map" in protein_2.metadata
        )

        if has_atom_data_1 and has_atom_data_2:
            # Use full atom-atom distances
            pos_pairs, neg_pairs = self._compute_interface_full_atom(
                protein_1.metadata["atom_pos"],
                protein_2.metadata["atom_pos"],
                protein_1.metadata["atom_res_map"],
                protein_2.metadata["atom_res_map"],
                threshold=self.interface_distance_graph,
            )
        else:
            # Fallback to CA-based distances
            logger.debug("Using CA-based distances for %s", system["id"])
            pos_pairs, neg_pairs = self._compute_interface(
                protein_1.graph.node_pos,
                protein_2.graph.node_pos,
                threshold=self.interface_distance_graph,
            )

        if pos_pairs.shape[1] < 5:
            logger.debug(
                "Too few interface pairs for graph: %d in %s",
                pos_pairs.shape[1],
                system["id"],
            )
            return None

        # Sample residue pairs
        idx_left, idx_right, labels = self._sample_pairs(pos_pairs, neg_pairs)
        if idx_left is None:
            # print(f"DEBUG: _sample_pairs returned None")
            return None

        # Compute and Sample Surface Pairs
        # Use simple cdist on surface points
        surf_pos_1 = getattr(protein_1.surface, "verts", None)
        if surf_pos_1 is None:
            # Surface generation failed, skip this sample
            return None

        surf_pos_2 = getattr(protein_2.surface, "verts", None)
        if surf_pos_2 is None:
            # Surface generation failed, skip this sample
            return None

        # Check minimum surface size
        if len(surf_pos_1) < 20 or len(surf_pos_2) < 20:
            # print(f"DEBUG: surfaces too small: {len(surf_pos_1)}/{len(surf_pos_2)}")
            return None

        # Surface interface uses same interface_distance
        surf_pos_pairs, surf_neg_pairs = self._compute_interface(
            surf_pos_1, surf_pos_2, threshold=self.interface_distance_surface
        )

        if surf_pos_pairs.shape[1] < 5:
            # Check if graphs are disconnected
            n_comp_1 = self._count_graph_components(protein_1.graph)
            n_comp_2 = self._count_graph_components(protein_2.graph)
            logger.debug(
                "Too few surface interface pairs: %d, verts: %d/%d, components: %d/%d, system: %s",
                surf_pos_pairs.shape[1],
                len(surf_pos_1),
                len(surf_pos_2),
                n_comp_1,
                n_comp_2,
                system["id"],
            )
            return None

        surf_idx_left, surf_idx_right, surf_labels = self._sample_pairs(
            surf_pos_pairs,
            surf_neg_pairs,
            neg_to_pos_ratio=getattr(self, "surface_neg_to_pos_ratio", 10.0),
            max_pos=-1,  # Always use all surface positives if possible
        )

        if surf_idx_left is None:
            # Fallback if no surface interaction found (rare but possible depending on threshold)
            # Create empty tensors to avoid crashing batching
            surf_idx_left = torch.tensor([], dtype=torch.long)
            surf_idx_right = torch.tensor([], dtype=torch.long)
            surf_labels = torch.tensor([], dtype=torch.float)

        # Validate residue indices
        if idx_left.max() >= g1_len or idx_right.max() >= g2_len:
            return None

        return Data(
            surface_1=protein_1.surface,
            graph_1=protein_1.graph,
            surface_2=protein_2.surface,
            graph_2=protein_2.graph,
            idx_left=idx_left,
            idx_right=idx_right,
            label=labels,
            surface_idx_left=surf_idx_left,
            surface_idx_right=surf_idx_right,
            surface_label=surf_labels,
            g1_len=g1_len,
            g2_len=g2_len,
            system_id=system["id"],
        )


class PinderAlignedDataset(PinderPairDataset):
    """
    Dataset for testing PINDER with alignment.

    Aligns the target protein (Apo or AF2) to the reference protein (Holo)
    based on sequence, and transfers the ground truth interface labels
    from Holo to the target.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_alignment_map(self, ref_protein, target_protein) -> Dict[int, int]:
        """Get a mapping from reference residue indices to target residue indices."""
        ref_seq_str, _ = self._get_sequence(ref_protein)
        tar_seq_str, _ = self._get_sequence(target_protein)

        if not ref_seq_str or not tar_seq_str:
            return {}

        ref_protein_seq = seq.ProteinSequence(ref_seq_str)
        tar_protein_seq = seq.ProteinSequence(tar_seq_str)

        matrix = align.SubstitutionMatrix.std_protein_matrix()  # BLOSUM62
        alignments = align.align_optimal(
            tar_protein_seq,
            ref_protein_seq,
            matrix,
            gap_penalty=(-10.0, -1.0),
            terminal_penalty=False,
            local=False,
            max_number=1,
        )

        if not alignments:
            return {}

        aln = alignments[0]
        trace = aln.trace
        mapping = {}
        for i in range(len(trace)):
            tar_idx = trace[i, 0]
            ref_idx = trace[i, 1]
            if tar_idx != -1 and ref_idx != -1:
                mapping[int(ref_idx)] = int(tar_idx)
        return mapping

    def _get_sequence(self, protein) -> Tuple[str, List[str]]:
        """
        Extract sequence and residue IDs from protein graph.
        Returns (sequence_string, list_of_res_ids)
        """
        if not hasattr(protein.graph, "node_names"):
            return None, None

        ids = protein.graph.node_names
        seq_chars = []
        for res_id in ids:
            # Format is typically "Chain:ResTypeResNum" or similar
            parts = res_id.split(":")
            if len(parts) < 2:
                seq_chars.append("X")
                continue

            # Extract ResType (could be 1-letter or 3-letter)
            # Match the leading letters (residue type) before any digits (residue number)
            match = re.search(r"([A-Za-z]+)", parts[1])
            if match:
                res_type = match.group(1)
                # use Biopython to handle 3-letter -> 1-letter if needed
                # seq1 can handle "GLY" -> "G", "G" -> "G", "MSE" -> "M", etc.
                res_code = SeqUtils.seq1(res_type)
            else:
                res_code = "X"

            seq_chars.append(res_code)

        seq = "".join(seq_chars)
        return seq, ids

    def _align_and_map(
        self,
        ref_protein,
        target_protein,
        ref_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map reference (Holo) indices to target (Apo) indices via alignment.
        """
        ref_to_tar = self._get_alignment_map(ref_protein, target_protein)

        # Extract mapped indices
        mapped = []
        for ridx in ref_indices.tolist():
            if ridx in ref_to_tar:
                mapped.append(ref_to_tar[ridx])

        if not mapped:
            return torch.tensor([], dtype=torch.long)
        return torch.tensor(mapped, dtype=torch.long)

    def __getitem__(self, idx: int) -> Optional[Data]:
        system = self.systems[idx]

        # 1. Load Reference (Holo)
        # Usually Holo is just the standard PDB ID.
        # The system ID is the Pinder ID.

        # 1. Load Target (Apo/AF2)
        target_r_path = self._get_pdb_path(system, "receptor")
        target_l_path = self._get_pdb_path(system, "ligand")

        target_r = self.protein_loader.load(
            f"{system['id']}_R_target", pdb_path=target_r_path, apply_noise=False
        )
        target_l = self.protein_loader.load(
            f"{system['id']}_L_target", pdb_path=target_l_path, apply_noise=False
        )

        if target_r is None or target_l is None:
            return None
        if not target_r.has_graph() or not target_l.has_graph():
            return None

        # 2. Check if we need alignment

        def get_holo_path(sys, ptype):
            path = sys.get(f"holo_{ptype}_path")
            if path:
                return path

            hid = sys.get(f"holo_{ptype}_id")
            if not hid:
                return None

            dummy = {"id": sys["id"], f"{ptype}_id": hid, "setting": "holo"}
            return self._get_pdb_path(dummy, ptype)

        holo_r_path = get_holo_path(system, "receptor")
        holo_l_path = get_holo_path(system, "ligand")

        if not holo_r_path or not holo_l_path:
            return None

        holo_r = self.protein_loader.load(
            f"{system['id']}_R_holo", pdb_path=holo_r_path, apply_noise=False
        )
        holo_l = self.protein_loader.load(
            f"{system['id']}_L_holo", pdb_path=holo_l_path, apply_noise=False
        )

        if holo_r is None or holo_l is None:
            return None
        if not holo_r.has_graph() or not holo_l.has_graph():
            return None

        # 3. Compute Interface on Holo (Reference)

        has_atom_data_1 = (
            "atom_pos" in holo_r.metadata and "atom_res_map" in holo_r.metadata
        )
        has_atom_data_2 = (
            "atom_pos" in holo_l.metadata and "atom_res_map" in holo_l.metadata
        )

        if has_atom_data_1 and has_atom_data_2:
            pos_pairs_ref, _ = self._compute_interface_full_atom(
                holo_r.metadata["atom_pos"],
                holo_l.metadata["atom_pos"],
                holo_r.metadata["atom_res_map"],
                holo_l.metadata["atom_res_map"],
                threshold=self.interface_distance_graph,
            )
        else:
            pos_pairs_ref, _ = self._compute_interface(
                holo_r.graph.node_pos,
                holo_l.graph.node_pos,
                threshold=self.interface_distance_graph,
            )

        if pos_pairs_ref.shape[1] < 1:
            return None

        # 4. Map Holo Indices -> Target Indices
        map_r = self._get_alignment_map(holo_r, target_r)
        map_l = self._get_alignment_map(holo_l, target_l)

        valid_pairs = []
        for i in range(pos_pairs_ref.shape[1]):
            u_ref = pos_pairs_ref[0, i]
            v_ref = pos_pairs_ref[1, i]

            if u_ref in map_r and v_ref in map_l:
                valid_pairs.append([map_r[u_ref], map_l[v_ref]])

        if not valid_pairs:
            return None

        pos_pairs = np.array(valid_pairs).T  # (2, K)

        # 5. Generate All potential Negatives for sampling (standard logic)
        n_res1 = len(target_r.graph.node_pos)
        n_res2 = len(target_l.graph.node_pos)
        neg_pairs = _generate_negative_pairs(pos_pairs, n_res1, n_res2)

        if neg_pairs.shape[1] == 0:
            return None

        # 6. Sample using parent logic (standardizes neg_to_pos_ratio behavior)
        idx_left, idx_right, labels = self._sample_pairs(pos_pairs, neg_pairs)
        if idx_left is None:
            return None

        # 7. Surfaces
        surf_idx_left = torch.tensor([], dtype=torch.long)
        surf_idx_right = torch.tensor([], dtype=torch.long)
        surf_labels = torch.tensor([], dtype=torch.float)

        return Data(
            surface_1=target_r.surface,
            graph_1=target_r.graph,
            surface_2=target_l.surface,
            graph_2=target_l.graph,
            idx_left=idx_left,
            idx_right=idx_right,
            label=labels,
            surface_idx_left=surf_idx_left,
            surface_idx_right=surf_idx_right,
            surface_label=surf_labels,
            g1_len=n_res1,
            g2_len=n_res2,
            system_id=system["id"],
        )
