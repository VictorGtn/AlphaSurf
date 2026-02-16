"""
MaSIF-Ligand dataset using the base protein infrastructure.
"""

import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.base import BaseProteinDataset

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

        data = protein.to_data()
        data.lig_coord = torch.from_numpy(lig_coord)
        data.label = lig_type

        return data
