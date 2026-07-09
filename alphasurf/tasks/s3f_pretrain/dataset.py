"""
CATH dataset for S3F-style masked residue prediction.

Uses the same CATH v4.3.0 S40 dompdb from HuggingFace (tyang816/cath) that
S3F and ProtSSN use. AlphaSurf generates CGAL alpha-complex surfaces
on-the-fly from the PDBs via ProteinLoader.

S3F's data pipeline:
  - All domains kept (no length filter at download time)
  - max_length=250 enforced by random crop at __getitem__ time (not filter)
  - Split ratios: [0.97, 0.02, 0.01] random on sorted file list
  - Masking: 15% of residues, 80% [MASK] / 10% random / 10% unchanged

Surface leakage prevention: at masked residues, sidechain atoms beyond Cb
are stripped from the coordinates BEFORE surface/graph generation. Both
branches see an Alanine-like structure (backbone + Cb only) at masked
positions, so no sidechain geometry leaks. Graph node features (one-hot,
hphob) are still zeroed separately by the model forward.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
from alphasurf.protein.graphs import res_type_idx_to_1
from alphasurf.protein.protein_loader import ProteinLoader
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

MASK_TYPE_MASK = 0
MASK_TYPE_RANDOM = 1
MASK_TYPE_UNCHANGED = 2


class CATHDataset(Dataset):
    """CATH dompdb dataset for masked residue prediction.

    __getitem__ returns a torch_geometric Data with:
      - graph: ResidueGraph (31-dim node features, coordinates already
        Ala-stripped at masked positions)
      - surface: SurfaceObject (built from Ala-stripped coordinates)
      - sequence: str (original AA sequence, cropped)
      - masked_positions: LongTensor (0-indexed into this protein's residues)
      - mask_types: LongTensor (0=mask, 1=random, 2=unchanged)
      - target_residues: LongTensor (original AA index in res_type_dict, 0-20)
      - random_aa_indices: LongTensor (random AA for type=1, -1 for others)
    """

    SPLIT_RATIOS = (0.97, 0.02, 0.01)

    def __init__(
        self,
        pdb_dir: str,
        split: str,
        protein_loader: ProteinLoader,
        mask_rate: float = 0.15,
        max_length: int = 250,
        k_surf_leak: int = 20,
        seed: int = 0,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        self.pdb_dir = pdb_dir
        self.split = split
        self.protein_loader = protein_loader
        self.mask_rate = mask_rate
        self.max_length = max_length
        self.k_surf_leak = k_surf_leak

        all_pdbs = sorted(os.listdir(pdb_dir))
        n = len(all_pdbs)
        n_train = int(n * self.SPLIT_RATIOS[0])
        n_val = int(n * self.SPLIT_RATIOS[1])
        if split == "train":
            self.pdbs = all_pdbs[:n_train]
        elif split == "val":
            self.pdbs = all_pdbs[n_train : n_train + n_val]
        else:
            self.pdbs = all_pdbs[n_train + n_val :]

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.pdbs)

    def __getitem__(self, idx: int) -> Optional[Data]:
        pdb_name = self.pdbs[idx]
        protein_name = pdb_name
        pdb_path = os.path.join(self.pdb_dir, pdb_name)

        crop_window = self._compute_crop_window(pdb_path)
        n_res_for_mask = (
            (crop_window[1] - crop_window[0])
            if crop_window is not None
            else self._count_residues_fast(pdb_path)
        )
        masked_positions = self._sample_positions(n_res_for_mask)

        try:
            protein = self.protein_loader.load(
                protein_name,
                pdb_path=pdb_path,
                crop_window=crop_window,
                ala_strip_positions=masked_positions.tolist(),
            )
        except Exception as e:
            logger.warning(f"Failed to load {pdb_name}: {e}")
            return None
        if protein is None:
            return None

        graph = protein.graph
        surface = protein.surface
        if graph is None or surface is None:
            return None

        n_res = graph.x.shape[0]
        n_surf = surface.x.shape[0] if surface.x is not None else 0
        if n_res < 32 or n_surf < 64:
            return None

        masked_positions = masked_positions[masked_positions < n_res]
        if len(masked_positions) == 0:
            return None

        sequence = self._extract_sequence(graph)
        if sequence is None or len(sequence) != n_res:
            return None

        mask_types, target_residues, random_aa_indices = self._complete_masking_plan(
            graph, masked_positions
        )

        return Data(
            graph=graph,
            surface=surface,
            sequence=sequence,
            masked_positions=masked_positions,
            mask_types=mask_types,
            target_residues=target_residues,
            random_aa_indices=random_aa_indices,
            protein_name=protein_name,
        )

    def _compute_crop_window(self, pdb_path: str):
        if self.max_length is None:
            return None
        n_res = self._count_residues_fast(pdb_path)
        if n_res <= self.max_length:
            return None
        start = int(self._rng.integers(0, n_res - self.max_length + 1))
        return (start, start + self.max_length)

    @staticmethod
    def _count_residues_fast(pdb_path: str) -> int:
        count = 0
        with open(pdb_path) as f:
            for line in f:
                if line[0:4] == "ATOM" and line[12:16].strip() == "CA":
                    count += 1
        return count

    def _sample_positions(self, n_res: int) -> torch.Tensor:
        n_mask = max(1, int(round(n_res * self.mask_rate)))
        n_mask = min(n_mask, n_res)
        positions = self._rng.choice(n_res, size=n_mask, replace=False)
        return torch.from_numpy(positions).long()

    def _complete_masking_plan(self, graph, positions: torch.Tensor):
        n_mask = len(positions)
        r = self._rng.random(n_mask)
        mask_types = np.where(
            r < 0.8,
            MASK_TYPE_MASK,
            np.where(r < 0.9, MASK_TYPE_RANDOM, MASK_TYPE_UNCHANGED),
        )
        mask_types = torch.from_numpy(mask_types).long()

        aa_idx = graph.x[:, 1:22].argmax(dim=-1).cpu().long()
        target_residues = aa_idx[positions]

        all_aa = list(range(20))
        random_aa = self._rng.choice(all_aa, size=n_mask)
        random_aa_indices = torch.from_numpy(random_aa).long()
        random_aa_indices = torch.where(
            mask_types == MASK_TYPE_RANDOM,
            random_aa_indices,
            torch.full_like(random_aa_indices, -1),
        )
        return mask_types, target_residues, random_aa_indices

    def _extract_sequence(self, graph) -> Optional[str]:
        try:
            aa_idx = graph.x[:, 1:22].argmax(dim=-1).cpu().numpy()
            return "".join(res_type_idx_to_1[i] for i in aa_idx)
        except Exception as e:
            logger.warning(f"Sequence extraction failed: {e}")
            return None
