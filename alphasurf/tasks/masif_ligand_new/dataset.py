"""
MaSIF-Ligand dataset using the base protein infrastructure.
"""

import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from alphasurf.protein.protein import Protein
from alphasurf.protein.protein_loader import ProteinLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data


class BaseProteinDataset(Dataset):
    """
    Base dataset handling protein loading with integrated transforms.

    This class provides the core data loading logic that can be reused
    across different tasks. Task-specific datasets inherit from this
    and override __getitem__ to add labels and task-specific processing.

    Transforms (noise, patch extraction) are applied DURING protein loading,
    not after, ensuring that computed features reflect the transformed geometry.
    """

    def __init__(
        self,
        names: List[str],
        protein_loader: ProteinLoader,
        apply_noise: bool = False,
        post_transforms: Optional[List[Callable[[Protein], Optional[Protein]]]] = None,
    ):
        """
        Args:
            names: List of protein/pocket names to load
            protein_loader: ProteinLoader instance (handles on-fly/disk loading)
            apply_noise: If True, apply noise augmentation during loading
            post_transforms: Optional transforms to apply AFTER loading
                           (for task-specific processing, not geometry changes)
        """
        self.names = names
        self.protein_loader = protein_loader
        self.apply_noise = apply_noise
        self.post_transforms = post_transforms or []

    def __len__(self) -> int:
        return len(self.names)

    def get_protein(self, idx: int) -> Optional[Protein]:
        """
        Load protein at index with transforms applied during loading.

        Args:
            idx: Dataset index

        Returns:
            Protein object or None if loading fails
        """
        name = self.names[idx]
        protein = self.protein_loader.load(name, apply_noise=self.apply_noise)

        if protein is None:
            return None

        for transform in self.post_transforms:
            protein = transform(protein)
            if protein is None:
                return None

        return protein

    def __getitem__(self, idx: int):
        """
        Get a protein at index.

        Override this in task-specific datasets to add labels and metadata.

        Returns:
            Protein.to_data() or None if loading fails
        """
        protein = self.get_protein(idx)
        if protein is None:
            return None
        return protein.to_data()


LIGAND_TYPES = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
LIGAND_TO_IDX = {lig: idx for idx, lig in enumerate(LIGAND_TYPES)}


def load_ligand_data(
    split_list_path: str,
    ligands_path: str,
    cache_path: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, int]]:
    """
    Load ligand coordinates and types for all pockets in a split.

    Args:
        split_list_path: Path to split file (e.g., train-list.txt)
        ligands_path: Path to directory with ligand coord/type files
        cache_path: Optional path to cache the loaded data

    Returns:
        Dict mapping pocket_name -> (ligand_coords, ligand_type_idx)
    """
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    all_pockets = {}
    split_list = open(split_list_path).read().splitlines()

    for pdb_chains in split_list:
        pdb = pdb_chains.split("_")[0]

        coords_file = os.path.join(ligands_path, f"{pdb}_ligand_coords.npy")
        types_file = os.path.join(ligands_path, f"{pdb}_ligand_types.npy")

        if not os.path.exists(coords_file) or not os.path.exists(types_file):
            continue

        ligand_coords = np.load(coords_file, allow_pickle=True, encoding="bytes")
        ligand_types = np.load(types_file)
        ligand_types = [
            lig.decode() if isinstance(lig, bytes) else lig for lig in ligand_types
        ]

        for idx, (lig_type, lig_coord) in enumerate(zip(ligand_types, ligand_coords)):
            if lig_type not in LIGAND_TO_IDX:
                continue
            pocket_name = f"{pdb_chains}_patch_{idx}_{lig_type}"
            all_pockets[pocket_name] = (
                lig_coord.astype(np.float32).reshape(-1, 3),
                LIGAND_TO_IDX[lig_type],
            )

    if cache_path is not None:
        with open(cache_path, "wb") as f:
            pickle.dump(all_pockets, f)

    return all_pockets


class MasifLigandDataset(BaseProteinDataset):
    """
    Dataset for MaSIF-Ligand task.

    Each sample is a protein binding site with:
    - Surface mesh (patch or full protein)
    - Residue graph
    - Ligand coordinates (for pooling)
    - Ligand type label (7 classes)
    """

    def __init__(
        self,
        pocket_data: Dict[str, Tuple[np.ndarray, int]],
        protein_loader: ProteinLoader,
        apply_noise: bool = False,
    ):
        """
        Args:
            pocket_data: Dict from load_ligand_data()
            protein_loader: Configured ProteinLoader instance
            apply_noise: Whether to apply noise augmentation
        """
        names = list(pocket_data.keys())
        super().__init__(names, protein_loader, apply_noise)
        self.pocket_data = pocket_data

    def __getitem__(self, idx: int) -> Optional[Data]:
        pocket_name = self.names[idx]
        lig_coord, lig_type = self.pocket_data[pocket_name]

        protein = self.get_protein(idx)
        if protein is None:
            return None
        if not protein.has_graph() or not protein.has_surface():
            return None

        data = protein.to_data()
        data.lig_coord = torch.from_numpy(lig_coord)
        data.label = lig_type

        return data
