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
import pandas as pd
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
    data_dir = Path(data_dir) if not isinstance(data_dir, Path) else data_dir

    # For test split, try setting-specific file first
    if split == "test" and test_setting:
        csv_path = data_dir / f"systems_{split}_{test_setting}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if max_systems is not None:
                df = df.iloc[:max_systems]
            systems = _df_to_systems(df)
            for s in systems:
                if not s.get("setting"):
                    s["setting"] = test_setting
            return systems

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
            r_path, l_path = _resolve_pinder_paths(
                system, split, test_setting, data_dir / "pdb"
            )

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
    system, split: str, setting: str = None, pdb_dir: Path = None
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve receptor and ligand paths from PinderSystem.

    For holo: writes native_R/native_L (from the dimer) to pdb_dir to avoid
    the corrupted test_set_pdbs normalization. For apo/af2: returns paths
    from the pinder package (those are in pdbs/ and are correct).
    """
    if split != "test":
        setting = "holo"
    if setting is None:
        setting = "holo"

    r_path = None
    l_path = None

    if setting == "holo":
        # native_R/L extracted from the native dimer — avoids corrupted
        # test_set_pdbs for test systems. Requires pdb_dir to write to.
        system_id = system.entry.id
        if pdb_dir is not None and system.native_R and system.native_L:
            pdb_dir = Path(pdb_dir)
            pdb_dir.mkdir(parents=True, exist_ok=True)
            suffix = f"_{setting}" if split == "test" else ""
            r_path = pdb_dir / f"{system_id}_R{suffix}.pdb"
            l_path = pdb_dir / f"{system_id}_L{suffix}.pdb"
            if not r_path.exists():
                system.native_R.to_pdb(r_path)
            if not l_path.exists():
                system.native_L.to_pdb(l_path)
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

    return r_path, l_path


def _df_to_systems(df) -> List[Dict]:
    """Convert DataFrame to list of system dicts."""
    systems = []
    for _, row in df.iterrows():
        system_id = row.get("id", str(row.name))
        entry = {
            "id": system_id,
            "receptor_id": row.get("receptor_id", f"{system_id}_R"),
            "ligand_id": row.get("ligand_id", f"{system_id}_L"),
            "setting": row.get("setting"),
            "split": row.get("split"),
            "receptor_path": row.get("receptor_path"),
            "ligand_path": row.get("ligand_path"),
        }
        if "cluster_id" in df.columns and pd.notna(row.get("cluster_id")):
            entry["cluster_id"] = row["cluster_id"]
        if "n_atoms_R" in df.columns and pd.notna(row.get("n_atoms_R")):
            entry["n_atoms_R"] = int(row["n_atoms_R"])
        if "n_atoms_L" in df.columns and pd.notna(row.get("n_atoms_L")):
            entry["n_atoms_L"] = int(row["n_atoms_L"])
        systems.append(entry)
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

    # "atom": map atom-level interface pairs to surface vertices at cache time
    # "full_atom": cache atom-level pairs, remap to vertices at __getitem__ time (survives mesh changes)
    # "residue": map residue-level interface pairs to surface vertices (via vert_atom_ids -> atom_res_map)
    # "dist": compute vertex-vertex distance pairs on surface
    SURFACE_LABEL_MODE = "atom"

    def __init__(
        self,
        systems: List[Dict],
        protein_loader: ProteinLoader,
        pdb_dir: str,
        neg_to_pos_ratio: float = 1.0,
        max_pos_per_pair: int = -1,
        interface_distance_graph: float = 8.0,
        interface_distance_surface: float = 8.0,
        # Legacy/Compatibility argument
        interface_distance: float = 8.0,
        surface_neg_to_pos_ratio: float = 10.0,
        interface_dir: Optional[str] = None,
        cluster_sampling: str = "single",
    ):
        """
        Args:
            systems: List of system dicts from load_pinder_split()
            protein_loader: ProteinLoader instance. If its noise_augmentor is
                enabled, __getitem__ noise is active and prepopulate_interface_cache
                caches clean interface pairs to be redrawn each draw.
            pdb_dir: Directory containing PDB files
            neg_to_pos_ratio: Ratio of negative to positive residue pairs to sample
            max_pos_per_pair: Maximum positive pairs per system (-1 for all)
            interface_distance_graph: Distance threshold (Å) for residue graph interface
            interface_distance_surface: Distance threshold (Å) for surface mesh interface
            interface_dir: Directory with precomputed interface pairs (disk mode)
            cluster_sampling: "single" treats each row as its own sample (current
                behavior). "multi" groups rows by cluster_id and samples one member
                per __getitem__ draw — requires cluster_id on each system dict.
        """
        self.systems = systems
        self.protein_loader = protein_loader
        self.pdb_dir = pdb_dir
        self.neg_to_pos_ratio = neg_to_pos_ratio
        self.max_pos_per_pair = max_pos_per_pair

        self.interface_distance_graph = interface_distance_graph
        self.interface_distance_surface = interface_distance_surface

        self.surface_neg_to_pos_ratio = surface_neg_to_pos_ratio
        self.interface_dir = interface_dir

        self._interface_cache = {}

        if cluster_sampling not in ("single", "multi"):
            raise ValueError(
                f"cluster_sampling must be 'single' or 'multi', got {cluster_sampling!r}"
            )
        self.cluster_sampling = cluster_sampling
        self._clusters: Optional[List[List[Dict]]] = None
        if cluster_sampling == "multi":
            self._build_cluster_index()

    def _build_cluster_index(self):
        """Group self.systems by cluster_id, preserving first-seen order."""
        self._clusters = []
        seen: Dict = {}
        missing = 0
        for sys in self.systems:
            cid = sys.get("cluster_id")
            if cid is None:
                missing += 1
                cid = sys["id"]
            if cid not in seen:
                seen[cid] = len(self._clusters)
                self._clusters.append([])
            self._clusters[seen[cid]].append(sys)
        if missing:
            logger.warning(
                "cluster_sampling='multi' but %d/%d systems lack cluster_id; "
                "treating each as its own cluster.",
                missing,
                len(self.systems),
            )

    @property
    def _noise_enabled(self) -> bool:
        augmentor = getattr(self.protein_loader, "noise_augmentor", None)
        return augmentor is not None and augmentor.enabled

    def prepopulate_interface_cache(self):
        """Pre-populate interface cache for all systems in the main process.

        Called before DataLoader spawns workers so that forked workers
        inherit a fully populated cache via copy-on-write, avoiding
        concurrent CGAL + operator generation in each worker.
        """
        if not self._noise_enabled:
            return
        if self.SURFACE_LABEL_MODE == "dist":
            # dist mode bypasses the interface cache entirely (recomputes on
            # the loaded sample each draw), so pre-population is wasted work.
            return

        print(
            f"[Cache] Pre-populating interface cache for {len(self.systems)} systems..."
        )
        cached = 0
        from_disk = 0
        for i, system in enumerate(self.systems):
            sid = system["id"]
            if sid in self._interface_cache:
                continue
            precomputed = self._load_precomputed_interface(sid)
            if precomputed is not None:
                self._interface_cache[sid] = {
                    k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                    for k, v in precomputed.items()
                }
                cached += 1
                from_disk += 1
                continue
            receptor_path = self._get_pdb_path(system, "receptor")
            ligand_path = self._get_pdb_path(system, "ligand")
            pairs = self._compute_clean_pairs(system, receptor_path, ligand_path)
            if pairs is not None:
                self._interface_cache[sid] = pairs
                cached += 1
            if (i + 1) % 500 == 0:
                print(
                    f"[Cache] {i + 1}/{len(self.systems)} processed ({cached} cached)"
                )
        print(
            f"[Cache] Done. {cached}/{len(self.systems)} systems cached "
            f"({from_disk} from disk)."
        )

    def __len__(self) -> int:
        if self._clusters is not None:
            return len(self._clusters)
        return len(self.systems)

    def _load_precomputed_interface(self, system_id: str):
        """Load precomputed interface pairs from disk."""
        if self.interface_dir is None:
            return None
        path = os.path.join(self.interface_dir, f"{system_id}.pt")
        if not os.path.exists(path):
            return None
        try:
            return torch.load(path, weights_only=True)
        except Exception:
            return None

    @staticmethod
    def _neg_pairs_from_complement(
        pos_pairs: np.ndarray, n1: int, n2: int
    ) -> np.ndarray:
        """Generate all (i, j) pairs NOT in pos_pairs, without cdist."""
        return _generate_negative_pairs(pos_pairs, n1, n2)

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

        # Also return raw atom pairs (for full_atom surface label mode)
        atom_pairs = torch.stack([atom_i, atom_j]).numpy()  # (2, M)

        return pos_pairs, neg_pairs, atom_pairs

    @staticmethod
    def _atom_pairs_to_vertex_pairs(
        atom_pos_pairs: np.ndarray,
        vert_atom_ids_1: np.ndarray,
        vert_atom_ids_2: np.ndarray,
    ) -> np.ndarray:
        """Map atom-level interface pairs to vertex-level pairs."""
        atom_to_vert_1 = {}
        for v, a in enumerate(vert_atom_ids_1):
            if a >= 0:
                atom_to_vert_1[int(a)] = v
        atom_to_vert_2 = {}
        for v, a in enumerate(vert_atom_ids_2):
            if a >= 0:
                atom_to_vert_2[int(a)] = v

        all_pairs = []
        for k in range(atom_pos_pairs.shape[1]):
            vi = atom_to_vert_1.get(int(atom_pos_pairs[0, k]))
            vj = atom_to_vert_2.get(int(atom_pos_pairs[1, k]))
            if vi is not None and vj is not None:
                all_pairs.append([vi, vj])

        if all_pairs:
            return np.array(all_pairs, dtype=np.int64).T
        return np.empty((2, 0), dtype=np.int64)

    @staticmethod
    def _residue_pairs_to_vertex_pairs(
        res_pairs: np.ndarray,
        vert_res_ids_1: np.ndarray,
        vert_res_ids_2: np.ndarray,
    ) -> np.ndarray:
        """Map residue-level interface pairs to vertex-level pairs."""
        res_to_vert_1 = {}
        for v, r in enumerate(vert_res_ids_1):
            if r >= 0:
                res_to_vert_1.setdefault(int(r), []).append(v)
        res_to_vert_2 = {}
        for v, r in enumerate(vert_res_ids_2):
            if r >= 0:
                res_to_vert_2.setdefault(int(r), []).append(v)

        all_pairs = []
        for k in range(res_pairs.shape[1]):
            vi_list = res_to_vert_1.get(int(res_pairs[0, k]), [])
            vj_list = res_to_vert_2.get(int(res_pairs[1, k]), [])
            for vi in vi_list:
                for vj in vj_list:
                    all_pairs.append([vi, vj])

        if all_pairs:
            return np.array(all_pairs, dtype=np.int64).T
        return np.empty((2, 0), dtype=np.int64)

    @staticmethod
    def _remap_atom_pairs_to_verts(
        atom_pairs: np.ndarray,
        surface_1,
        surface_2,
    ) -> np.ndarray:
        """Remap cached atom-level pairs to vertex indices on actual surfaces.

        Unlike _atom_pairs_to_vertex_pairs which takes pre-extracted vert_atom_ids arrays,
        this extracts vert_atom_ids directly from surface objects, making it suitable
        for deferred mapping (e.g. on noised surfaces).
        """
        vaid_1 = getattr(surface_1, "vert_atom_ids", None)
        vaid_2 = getattr(surface_2, "vert_atom_ids", None)
        if vaid_1 is None or vaid_2 is None:
            return np.empty((2, 0), dtype=np.int64)

        aid_1 = vaid_1.numpy() if isinstance(vaid_1, torch.Tensor) else vaid_1
        aid_2 = vaid_2.numpy() if isinstance(vaid_2, torch.Tensor) else vaid_2

        atom_to_vert_1 = {}
        for v, a in enumerate(aid_1):
            if a >= 0:
                atom_to_vert_1[int(a)] = v
        atom_to_vert_2 = {}
        for v, a in enumerate(aid_2):
            if a >= 0:
                atom_to_vert_2[int(a)] = v

        all_pairs = []
        for k in range(atom_pairs.shape[1]):
            vi = atom_to_vert_1.get(int(atom_pairs[0, k]))
            vj = atom_to_vert_2.get(int(atom_pairs[1, k]))
            if vi is not None and vj is not None:
                all_pairs.append([vi, vj])

        if all_pairs:
            return np.array(all_pairs, dtype=np.int64).T
        return np.empty((2, 0), dtype=np.int64)

    def _sample_pairs_with_lazy_neg(
        self,
        pos_pairs: np.ndarray,
        n1: int,
        n2: int,
        neg_to_pos_ratio: float = None,
        max_pos: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample positive pairs and generate negative indices on-the-fly."""
        ratio = (
            neg_to_pos_ratio if neg_to_pos_ratio is not None else self.neg_to_pos_ratio
        )
        max_p = max_pos if max_pos is not None else self.max_pos_per_pair

        num_pos = pos_pairs.shape[1]
        if num_pos == 0:
            return None, None, None

        num_pos_use = (
            min(num_pos, max(1, int(num_pos / ratio)))
            if max_p <= 0
            else min(num_pos, max_p, int(num_pos / ratio))
        )
        num_pos_use = max(1, int(math.ceil(num_pos_use)))
        num_neg_use = max(1, int(math.ceil(num_pos_use * ratio)))

        pos_idx = np.random.choice(num_pos, size=num_pos_use, replace=False)
        pos_sampled = pos_pairs[:, pos_idx]

        # Generate negatives excluding positive pairs
        pos_set = set(zip(pos_pairs[0].tolist(), pos_pairs[1].tolist()))
        neg_i_list, neg_j_list = [], []
        oversample = max(num_neg_use * 3, 1000)
        while len(neg_i_list) < num_neg_use:
            cand_i = np.random.randint(0, n1, size=oversample)
            cand_j = np.random.randint(0, n2, size=oversample)
            for i, j in zip(cand_i.tolist(), cand_j.tolist()):
                if (i, j) not in pos_set:
                    neg_i_list.append(i)
                    neg_j_list.append(j)
                    if len(neg_i_list) >= num_neg_use:
                        break
        neg_i = np.array(neg_i_list[:num_neg_use])
        neg_j = np.array(neg_j_list[:num_neg_use])

        idx_left = torch.from_numpy(np.concatenate([pos_sampled[0], neg_i]))
        idx_right = torch.from_numpy(np.concatenate([pos_sampled[1], neg_j]))
        labels = torch.cat(
            [
                torch.ones(num_pos_use),
                torch.zeros(num_neg_use),
            ]
        )

        return idx_left, idx_right, labels

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

    def _compute_clean_pairs(self, system, receptor_path, ligand_path):
        """Load clean proteins and compute interface pairs for caching."""
        receptor_name = system.get("receptor_id", f"{system['id']}_R")
        ligand_name = system.get("ligand_id", f"{system['id']}_L")

        clean_1 = self.protein_loader.load_clean(receptor_name, pdb_path=receptor_path)
        clean_2 = self.protein_loader.load_clean(ligand_name, pdb_path=ligand_path)

        if clean_1 is None or clean_2 is None:
            return None
        if not clean_1.has_graph() or not clean_2.has_graph():
            return None

        g1_len = len(clean_1.graph.node_pos)
        g2_len = len(clean_2.graph.node_pos)
        if g1_len < 20 or g2_len < 20:
            return None

        # Graph interface
        assert (
            "atom_pos" in clean_1.metadata and "atom_res_map" in clean_1.metadata
        ), f"Missing atom data for {system['id']} receptor"
        assert (
            "atom_pos" in clean_2.metadata and "atom_res_map" in clean_2.metadata
        ), f"Missing atom data for {system['id']} ligand"

        pos_pairs, neg_pairs, atom_pairs = self._compute_interface_full_atom(
            clean_1.metadata["atom_pos"],
            clean_2.metadata["atom_pos"],
            clean_1.metadata["atom_res_map"],
            clean_2.metadata["atom_res_map"],
            threshold=self.interface_distance_graph,
        )

        if pos_pairs.shape[1] < 5:
            return None

        if clean_1.surface is None or clean_2.surface is None:
            return None

        # Surface interface — cache positive pairs + dims, generate negatives on-the-fly
        surf_pos_pairs = np.empty((2, 0))
        surf_atom_pairs = None
        surf_n1 = 0
        surf_n2 = 0

        if self.SURFACE_LABEL_MODE == "residue":
            assert hasattr(
                clean_1.surface, "vert_atom_ids"
            ), f"{system['id']}: residue mode requires vert_atom_ids on surface"
            assert hasattr(
                clean_2.surface, "vert_atom_ids"
            ), f"{system['id']}: residue mode requires vert_atom_ids on surface"
            assert (
                clean_1.metadata and "atom_res_map" in clean_1.metadata
            ), f"{system['id']}: residue mode requires atom_res_map in metadata"
            assert (
                clean_2.metadata and "atom_res_map" in clean_2.metadata
            ), f"{system['id']}: residue mode requires atom_res_map in metadata"

            vert_aid_1 = clean_1.surface.vert_atom_ids
            vert_aid_2 = clean_2.surface.vert_atom_ids
            arm_1 = clean_1.metadata["atom_res_map"]
            arm_2 = clean_2.metadata["atom_res_map"]

            assert (
                vert_aid_1 >= 0
            ).any(), f"{system['id']}: receptor has no vert-atom matches"
            assert (
                vert_aid_2 >= 0
            ).any(), f"{system['id']}: ligand has no vert-atom matches"

            aid_1 = (
                vert_aid_1.numpy()
                if isinstance(vert_aid_1, torch.Tensor)
                else vert_aid_1
            )
            aid_2 = (
                vert_aid_2.numpy()
                if isinstance(vert_aid_2, torch.Tensor)
                else vert_aid_2
            )
            arm_1_np = arm_1.numpy() if isinstance(arm_1, torch.Tensor) else arm_1
            arm_2_np = arm_2.numpy() if isinstance(arm_2, torch.Tensor) else arm_2

            valid_1 = aid_1 >= 0
            valid_2 = aid_2 >= 0
            vrid_1 = np.full_like(aid_1, -1)
            vrid_2 = np.full_like(aid_2, -1)
            vrid_1[valid_1] = arm_1_np[aid_1[valid_1]]
            vrid_2[valid_2] = arm_2_np[aid_2[valid_2]]

            surf_pos_pairs = self._residue_pairs_to_vertex_pairs(
                pos_pairs, vrid_1, vrid_2
            )
            surf_n1 = len(clean_1.surface.verts)
            surf_n2 = len(clean_2.surface.verts)
        elif self.SURFACE_LABEL_MODE == "atom":
            assert hasattr(
                clean_1.surface, "vert_atom_ids"
            ), f"{system['id']}: atom mode requires vert_atom_ids on surface"
            assert hasattr(
                clean_2.surface, "vert_atom_ids"
            ), f"{system['id']}: atom mode requires vert_atom_ids on surface"
            assert (
                clean_1.metadata and "atom_pos" in clean_1.metadata
            ), f"{system['id']}: atom mode requires atom_pos in metadata"
            assert (
                clean_2.metadata and "atom_pos" in clean_2.metadata
            ), f"{system['id']}: atom mode requires atom_pos in metadata"

            vert_aid_1 = clean_1.surface.vert_atom_ids
            vert_aid_2 = clean_2.surface.vert_atom_ids
            atom_pos_1 = clean_1.metadata["atom_pos"]
            atom_pos_2 = clean_2.metadata["atom_pos"]

            assert (
                vert_aid_1 >= 0
            ).any(), f"{system['id']}: receptor has no vert-atom matches"
            assert (
                vert_aid_2 >= 0
            ).any(), f"{system['id']}: ligand has no vert-atom matches"

            atom_pos_pairs, _ = self._compute_interface(
                atom_pos_1, atom_pos_2, threshold=self.interface_distance_surface
            )
            aid_1 = (
                vert_aid_1.numpy()
                if isinstance(vert_aid_1, torch.Tensor)
                else vert_aid_1
            )
            aid_2 = (
                vert_aid_2.numpy()
                if isinstance(vert_aid_2, torch.Tensor)
                else vert_aid_2
            )
            surf_pos_pairs = self._atom_pairs_to_vertex_pairs(
                atom_pos_pairs, aid_1, aid_2
            )
            surf_n1 = len(clean_1.surface.verts)
            surf_n2 = len(clean_2.surface.verts)
        elif self.SURFACE_LABEL_MODE == "full_atom":
            assert (
                clean_1.metadata and "atom_pos" in clean_1.metadata
            ), f"{system['id']}: full_atom mode requires atom_pos in metadata"
            assert (
                clean_2.metadata and "atom_pos" in clean_2.metadata
            ), f"{system['id']}: full_atom mode requires atom_pos in metadata"
            assert hasattr(
                clean_1.surface, "vert_atom_ids"
            ), f"{system['id']}: full_atom mode requires vert_atom_ids on surface"
            assert hasattr(
                clean_2.surface, "vert_atom_ids"
            ), f"{system['id']}: full_atom mode requires vert_atom_ids on surface"
            # Compute atom pairs from surface vertex distances (not graph atom distances)
            surf_v1 = clean_1.surface.verts
            surf_v2 = clean_2.surface.verts
            if not isinstance(surf_v1, np.ndarray):
                surf_v1 = np.asarray(surf_v1)
            if not isinstance(surf_v2, np.ndarray):
                surf_v2 = np.asarray(surf_v2)
            if len(surf_v1) >= 20 and len(surf_v2) >= 20:
                vtx_dists = torch.cdist(
                    torch.from_numpy(surf_v1).float(),
                    torch.from_numpy(surf_v2).float(),
                )
                vi, vj = torch.where(vtx_dists < self.interface_distance_surface)
                vaid_1 = clean_1.surface.vert_atom_ids
                vaid_2 = clean_2.surface.vert_atom_ids
                vaid_1 = vaid_1.numpy() if isinstance(vaid_1, torch.Tensor) else vaid_1
                vaid_2 = vaid_2.numpy() if isinstance(vaid_2, torch.Tensor) else vaid_2
                _pairs = []
                for k in range(len(vi)):
                    a1 = vaid_1[vi[k].item()]
                    a2 = vaid_2[vj[k].item()]
                    if a1 >= 0 and a2 >= 0:
                        _pairs.append([a1, a2])
                if _pairs:
                    surf_atom_pairs = np.unique(
                        np.array(_pairs, dtype=np.int64).T, axis=1
                    )
                else:
                    surf_atom_pairs = np.empty((2, 0), dtype=np.int64)
            else:
                surf_atom_pairs = np.empty((2, 0), dtype=np.int64)
            surf_n1 = len(surf_v1)
            surf_n2 = len(surf_v2)
        elif self.SURFACE_LABEL_MODE == "dist":
            surf_pos_1 = clean_1.surface.verts
            surf_pos_2 = clean_2.surface.verts
            surf_pos_pairs, _ = self._compute_interface(
                surf_pos_1, surf_pos_2, threshold=self.interface_distance_surface
            )
            surf_n1 = len(clean_1.surface.verts)
            surf_n2 = len(clean_2.surface.verts)

        return {
            "pos_pairs": pos_pairs,
            "neg_pairs": neg_pairs,
            "surf_pos_pairs": surf_pos_pairs,
            "surf_atom_pairs": surf_atom_pairs,
            "surf_n1": surf_n1,
            "surf_n2": surf_n2,
        }

    def __getitem__(self, idx: int) -> Optional[Data]:
        if self._clusters is not None:
            cluster = self._clusters[idx]
            system = cluster[np.random.randint(len(cluster))]
        else:
            system = self.systems[idx]

        # Resolve paths
        receptor_path = self._get_pdb_path(system, "receptor")
        ligand_path = self._get_pdb_path(system, "ligand")

        # Get interface pairs (cached on clean structures when noise is active)
        # dist mode bypasses the cache entirely and recomputes on the loaded sample.
        if self._noise_enabled and self.SURFACE_LABEL_MODE != "dist":
            sid = system["id"]
            if sid not in self._interface_cache:
                precomputed = self._load_precomputed_interface(sid)
                if precomputed is not None:
                    self._interface_cache[sid] = {
                        k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                        for k, v in precomputed.items()
                    }
                else:
                    pairs = self._compute_clean_pairs(
                        system, receptor_path, ligand_path
                    )
                    if pairs is None:
                        return None
                    self._interface_cache[sid] = pairs
            pairs = self._interface_cache[sid]
            pos_pairs = pairs["pos_pairs"]
            neg_pairs = pairs["neg_pairs"]

        # Load proteins (with noise for training, without for eval)
        # In disk mode, test systems use setting suffix for precomputed filenames
        setting = system.get("setting")
        receptor_name = system.get("receptor_id", f"{system['id']}_R")
        ligand_name = system.get("ligand_id", f"{system['id']}_L")
        if setting:
            receptor_name = f"{receptor_name}_{setting}"
            ligand_name = f"{ligand_name}_{setting}"
        protein_1 = self.protein_loader.load(receptor_name, pdb_path=receptor_path)
        protein_2 = self.protein_loader.load(ligand_name, pdb_path=ligand_path)

        if protein_1 is None or protein_2 is None:
            return None

        if not protein_1.has_graph() or not protein_2.has_graph():
            return None

        if protein_1.surface is None or protein_2.surface is None:
            return None

        g1_len = len(protein_1.graph.node_pos)
        g2_len = len(protein_2.graph.node_pos)

        if g1_len < 20 or g2_len < 20:
            return None

        # --- Graph residue pairs ---
        precomputed = None
        if self.SURFACE_LABEL_MODE == "dist":
            # dist mode: always compute fresh from the currently-loaded proteins.
            # Under noise, metadata["atom_pos"] reflects the noised atoms (loader Edit).
            assert (
                "atom_pos" in protein_1.metadata
                and "atom_res_map" in protein_1.metadata
            ), f"Missing atom data for {system['id']} receptor"
            assert (
                "atom_pos" in protein_2.metadata
                and "atom_res_map" in protein_2.metadata
            ), f"Missing atom data for {system['id']} ligand"

            pos_pairs, neg_pairs, _ = self._compute_interface_full_atom(
                protein_1.metadata["atom_pos"],
                protein_2.metadata["atom_pos"],
                protein_1.metadata["atom_res_map"],
                protein_2.metadata["atom_res_map"],
                threshold=self.interface_distance_graph,
            )

            if pos_pairs.shape[1] < 5:
                return None
        elif self._noise_enabled:
            # Cache already populated by the early block above
            pairs = self._interface_cache[sid]
            pos_pairs = pairs["pos_pairs"]
            neg_pairs = pairs["neg_pairs"]
        else:
            precomputed = self._load_precomputed_interface(system["id"])

            if precomputed is not None:
                pos_pairs = precomputed["graph_pos_pairs"].numpy()
                neg_pairs = _generate_negative_pairs(
                    pos_pairs,
                    int(precomputed["graph_n_res1"]),
                    int(precomputed["graph_n_res2"]),
                )

                if pos_pairs.shape[1] < 5:
                    return None
            else:
                assert (
                    "atom_pos" in protein_1.metadata
                    and "atom_res_map" in protein_1.metadata
                ), f"Missing atom data for {system['id']} receptor"
                assert (
                    "atom_pos" in protein_2.metadata
                    and "atom_res_map" in protein_2.metadata
                ), f"Missing atom data for {system['id']} ligand"

                pos_pairs, neg_pairs, atom_pairs = self._compute_interface_full_atom(
                    protein_1.metadata["atom_pos"],
                    protein_2.metadata["atom_pos"],
                    protein_1.metadata["atom_res_map"],
                    protein_2.metadata["atom_res_map"],
                    threshold=self.interface_distance_graph,
                )

                if pos_pairs.shape[1] < 5:
                    return None

        # --- Surface vertex pairs ---
        surf_pos_pairs = np.empty((2, 0))
        surf_neg_pairs = np.empty((2, 0))
        surf_n1 = 0
        surf_n2 = 0

        if self.SURFACE_LABEL_MODE == "dist":
            # dist mode: always recompute on currently-loaded verts (clean or noised).
            surf_pos_1 = protein_1.surface.verts
            surf_pos_2 = protein_2.surface.verts
            if len(surf_pos_1) >= 20 and len(surf_pos_2) >= 20:
                surf_pos_pairs, _ = self._compute_interface(
                    surf_pos_1,
                    surf_pos_2,
                    threshold=self.interface_distance_surface,
                )
            surf_n1 = len(surf_pos_1)
            surf_n2 = len(surf_pos_2)
        elif self._noise_enabled:
            cached_surf = self._interface_cache.get(system["id"], {})
            if self.SURFACE_LABEL_MODE == "full_atom":
                surf_atom_pairs = cached_surf.get("surf_atom_pairs")
                if surf_atom_pairs is not None:
                    surf_pos_pairs = self._remap_atom_pairs_to_verts(
                        surf_atom_pairs, protein_1.surface, protein_2.surface
                    )
                surf_n1 = len(protein_1.surface.verts)
                surf_n2 = len(protein_2.surface.verts)
            elif self.SURFACE_LABEL_MODE == "residue":
                # Remap cached residue pairs to noised surface vertices
                assert hasattr(
                    protein_1.surface, "vert_atom_ids"
                ), f"{system['id']}: residue mode requires vert_atom_ids"
                assert hasattr(
                    protein_2.surface, "vert_atom_ids"
                ), f"{system['id']}: residue mode requires vert_atom_ids"
                assert (
                    protein_1.metadata and "atom_res_map" in protein_1.metadata
                ), f"{system['id']}: residue mode requires atom_res_map"
                assert (
                    protein_2.metadata and "atom_res_map" in protein_2.metadata
                ), f"{system['id']}: residue mode requires atom_res_map"

                vert_aid_1 = protein_1.surface.vert_atom_ids
                vert_aid_2 = protein_2.surface.vert_atom_ids
                arm_1 = protein_1.metadata["atom_res_map"]
                arm_2 = protein_2.metadata["atom_res_map"]

                aid_1 = (
                    vert_aid_1.numpy()
                    if isinstance(vert_aid_1, torch.Tensor)
                    else vert_aid_1
                )
                aid_2 = (
                    vert_aid_2.numpy()
                    if isinstance(vert_aid_2, torch.Tensor)
                    else vert_aid_2
                )
                arm_1_np = arm_1.numpy() if isinstance(arm_1, torch.Tensor) else arm_1
                arm_2_np = arm_2.numpy() if isinstance(arm_2, torch.Tensor) else arm_2

                valid_1 = aid_1 >= 0
                valid_2 = aid_2 >= 0
                vrid_1 = np.full_like(aid_1, -1)
                vrid_2 = np.full_like(aid_2, -1)
                vrid_1[valid_1] = arm_1_np[aid_1[valid_1]]
                vrid_2[valid_2] = arm_2_np[aid_2[valid_2]]

                surf_pos_pairs = self._residue_pairs_to_vertex_pairs(
                    cached_surf["pos_pairs"], vrid_1, vrid_2
                )
                surf_n1 = len(protein_1.surface.verts)
                surf_n2 = len(protein_2.surface.verts)
            else:
                surf_pos_pairs = cached_surf.get("surf_pos_pairs", np.empty((2, 0)))
                surf_n1 = cached_surf.get("surf_n1", 0)
                surf_n2 = cached_surf.get("surf_n2", 0)
        elif precomputed is not None:
            surf_pos_pairs = precomputed["surf_pos_pairs"].numpy()
            surf_n1 = int(precomputed["surf_n1"])
            surf_n2 = int(precomputed["surf_n2"])
            if surf_pos_pairs.shape[1] > 0 and surf_n1 > 0 and surf_n2 > 0:
                surf_neg_pairs = self._neg_pairs_from_complement(
                    surf_pos_pairs, surf_n1, surf_n2
                )
        else:
            atom_pos_1 = (
                protein_1.metadata.get("atom_pos") if protein_1.metadata else None
            )
            atom_pos_2 = (
                protein_2.metadata.get("atom_pos") if protein_2.metadata else None
            )

            if self.SURFACE_LABEL_MODE == "residue":
                assert hasattr(
                    protein_1.surface, "vert_atom_ids"
                ), f"{system['id']}: residue mode requires vert_atom_ids"
                assert hasattr(
                    protein_2.surface, "vert_atom_ids"
                ), f"{system['id']}: residue mode requires vert_atom_ids"
                assert (
                    protein_1.metadata and "atom_res_map" in protein_1.metadata
                ), f"{system['id']}: residue mode requires atom_res_map"
                assert (
                    protein_2.metadata and "atom_res_map" in protein_2.metadata
                ), f"{system['id']}: residue mode requires atom_res_map"

                vert_aid_1 = protein_1.surface.vert_atom_ids
                vert_aid_2 = protein_2.surface.vert_atom_ids
                arm_1 = protein_1.metadata["atom_res_map"]
                arm_2 = protein_2.metadata["atom_res_map"]

                aid_1 = (
                    vert_aid_1.numpy()
                    if isinstance(vert_aid_1, torch.Tensor)
                    else vert_aid_1
                )
                aid_2 = (
                    vert_aid_2.numpy()
                    if isinstance(vert_aid_2, torch.Tensor)
                    else vert_aid_2
                )
                arm_1_np = arm_1.numpy() if isinstance(arm_1, torch.Tensor) else arm_1
                arm_2_np = arm_2.numpy() if isinstance(arm_2, torch.Tensor) else arm_2

                valid_1 = aid_1 >= 0
                valid_2 = aid_2 >= 0
                vrid_1 = np.full_like(aid_1, -1)
                vrid_2 = np.full_like(aid_2, -1)
                vrid_1[valid_1] = arm_1_np[aid_1[valid_1]]
                vrid_2[valid_2] = arm_2_np[aid_2[valid_2]]

                surf_pos_pairs = self._residue_pairs_to_vertex_pairs(
                    pos_pairs, vrid_1, vrid_2
                )
                surf_n1 = len(protein_1.surface.verts)
                surf_n2 = len(protein_2.surface.verts)

            elif self.SURFACE_LABEL_MODE == "atom":
                assert hasattr(
                    protein_1.surface, "vert_atom_ids"
                ), f"{system['id']}: atom mode requires vert_atom_ids"
                assert hasattr(
                    protein_2.surface, "vert_atom_ids"
                ), f"{system['id']}: atom mode requires vert_atom_ids"
                assert (
                    atom_pos_1 is not None
                ), f"{system['id']}: atom mode requires atom_pos"
                assert (
                    atom_pos_2 is not None
                ), f"{system['id']}: atom mode requires atom_pos"

                vert_aid_1 = protein_1.surface.vert_atom_ids
                vert_aid_2 = protein_2.surface.vert_atom_ids

                atom_pos_pairs, _ = self._compute_interface(
                    atom_pos_1,
                    atom_pos_2,
                    threshold=self.interface_distance_surface,
                )
                aid_1 = (
                    vert_aid_1.numpy()
                    if isinstance(vert_aid_1, torch.Tensor)
                    else vert_aid_1
                )
                aid_2 = (
                    vert_aid_2.numpy()
                    if isinstance(vert_aid_2, torch.Tensor)
                    else vert_aid_2
                )
                surf_pos_pairs = self._atom_pairs_to_vertex_pairs(
                    atom_pos_pairs,
                    aid_1,
                    aid_2,
                )
                surf_n1 = len(protein_1.surface.verts)
                surf_n2 = len(protein_2.surface.verts)

            elif self.SURFACE_LABEL_MODE == "full_atom":
                assert (
                    atom_pos_1 is not None
                ), f"{system['id']}: full_atom mode requires atom_pos"
                assert (
                    atom_pos_2 is not None
                ), f"{system['id']}: full_atom mode requires atom_pos"

                # Reuse atom pairs from graph interface (same threshold)
                surf_pos_pairs = self._remap_atom_pairs_to_verts(
                    atom_pairs, protein_1.surface, protein_2.surface
                )
                surf_n1 = len(protein_1.surface.verts)
                surf_n2 = len(protein_2.surface.verts)

        # Sample graph pairs
        idx_left, idx_right, labels = self._sample_pairs(pos_pairs, neg_pairs)
        if idx_left is None:
            return None

        # Sample surface pairs
        if surf_pos_pairs.shape[1] >= 5:
            use_lazy_neg = surf_neg_pairs.shape[1] == 0
            if use_lazy_neg:
                # Noise/residue mode: generate surface negatives on-the-fly
                surf_idx_left, surf_idx_right, surf_labels = (
                    self._sample_pairs_with_lazy_neg(
                        surf_pos_pairs,
                        surf_n1,
                        surf_n2,
                        neg_to_pos_ratio=getattr(
                            self, "surface_neg_to_pos_ratio", 10.0
                        ),
                        max_pos=-1,
                    )
                )
            else:
                surf_idx_left, surf_idx_right, surf_labels = self._sample_pairs(
                    surf_pos_pairs,
                    surf_neg_pairs,
                    neg_to_pos_ratio=getattr(self, "surface_neg_to_pos_ratio", 10.0),
                    max_pos=-1,
                )
            if surf_idx_left is None:
                surf_idx_left = torch.tensor([], dtype=torch.long)
                surf_idx_right = torch.tensor([], dtype=torch.long)
                surf_labels = torch.tensor([], dtype=torch.float)
        else:
            surf_idx_left = torch.tensor([], dtype=torch.long)
            surf_idx_right = torch.tensor([], dtype=torch.long)
            surf_labels = torch.tensor([], dtype=torch.float)

        # Validate residue indices
        if idx_left.max() >= g1_len or idx_right.max() >= g2_len:
            return None

        # Validate surface vertex indices (joint noise can change vertex count)
        s1_len = len(protein_1.surface.verts)
        s2_len = len(protein_2.surface.verts)
        if (
            surf_idx_left.numel() > 0
            and surf_idx_right.numel() > 0
            and (surf_idx_left.max() >= s1_len or surf_idx_right.max() >= s2_len)
        ):
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
                # If already 1-letter, use directly; otherwise convert 3-letter -> 1-letter
                if len(res_type) == 1:
                    res_code = res_type
                else:
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

        setting = system.get("setting", "holo")
        target_r = self.protein_loader.load(
            f"{system['id']}_R_{setting}", pdb_path=target_r_path
        )
        target_l = self.protein_loader.load(
            f"{system['id']}_L_{setting}", pdb_path=target_l_path
        )

        if target_r is None or target_l is None:
            return None
        if not target_r.has_graph() or not target_l.has_graph():
            return None
        if target_r.surface is None or target_l.surface is None:
            return None

        # 2. Check if we need alignment

        def get_holo_path(sys, ptype):
            path = sys.get(f"holo_{ptype}_path")
            if path:
                return path

            hid = sys.get(f"holo_{ptype}_id")
            if not hid:
                suffix = "R" if ptype == "receptor" else "L"
                hid = f"{sys['id']}_{suffix}"

            dummy = {"id": sys["id"], f"{ptype}_id": hid, "setting": "holo"}
            return self._get_pdb_path(dummy, ptype)

        holo_r_path = get_holo_path(system, "receptor")
        holo_l_path = get_holo_path(system, "ligand")

        if not holo_r_path or not holo_l_path:
            return None

        holo_r = self.protein_loader.load(
            f"{system['id']}_R_holo", pdb_path=holo_r_path
        )
        holo_l = self.protein_loader.load(
            f"{system['id']}_L_holo", pdb_path=holo_l_path
        )

        if holo_r is None or holo_l is None:
            return None
        if not holo_r.has_graph() or not holo_l.has_graph():
            return None

        # 3. Compute Interface on Holo (Reference)

        assert (
            "atom_pos" in holo_r.metadata and "atom_res_map" in holo_r.metadata
        ), f"Missing atom data for {system['id']} holo receptor"
        assert (
            "atom_pos" in holo_l.metadata and "atom_res_map" in holo_l.metadata
        ), f"Missing atom data for {system['id']} holo ligand"

        pos_pairs_ref, _, _ = self._compute_interface_full_atom(
            holo_r.metadata["atom_pos"],
            holo_l.metadata["atom_pos"],
            holo_r.metadata["atom_res_map"],
            holo_l.metadata["atom_res_map"],
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
