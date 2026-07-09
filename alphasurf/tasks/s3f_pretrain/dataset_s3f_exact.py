"""
CATH dataset for S3F-exact encoder using precomputed dMaSIF point clouds.

Loads .pt files produced by precompute_s3f_exact.py. Each file contains:
  - sequence: str
  - ca_pos: (n_res, 3)
  - bb_pos: (n_res, 3, 3) — N/CA/C per residue
  - surf_pos: (M, 3)
  - surf_normals: (M, 3)
  - surf_feat: (M, 42) — curvature(10) + HKS(32)

Per-epoch operations:
  - Random crop to [0, max_length) if longer
  - Filter surface points whose 3-NN residue mean falls in crop
  - Build Cα radius=10 graph + RBF(D_max=20) edges
  - Build surface kNN k=16 graph + RBF edges
  - Compute res2surf (60 NN per residue: 20 per backbone atom × 3)
  - Sample masked residues (15%, 80/10/10 split)
  - Build torch_geometric Data with everything S3FPretrainNet needs

Masking (ESM tokens + residue_feature zero) is applied in the model forward,
not here — same convention as the AlphaSurf-pipeline CATHDataset.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected


class S3FSurfaceData(Data):
    """Surface Data with batching-aware index attributes.

    PyG's default Batch only offsets well-known keys (edge_index, etc).
    Custom Long tensors like `res2surf` (indices into surface nodes) need
    an explicit `__inc__` so they get offset by num_nodes during batching.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "res2surf":
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "res2surf":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


logger = logging.getLogger(__name__)

MASK_TYPE_MASK = 0
MASK_TYPE_RANDOM = 1
MASK_TYPE_UNCHANGED = 2

RADIUS_CUTOFF = 10.0
RBF_D_MAX = 20.0
RBF_DIM = 16
SURF_KNN = 16
FUSION_K_PER_ATOM = 20
SURF_INIT_K = 3


def _rbf(d, d_min=0.0, d_max=RBF_D_MAX, d_count=RBF_DIM):
    from alphasurf.network_utils.communication.passing_utils import _rbf as _impl

    return _impl(d, D_min=d_min, D_max=d_max, D_count=d_count)


def _radius_edges(pos, cutoff=RADIUS_CUTOFF):
    from scipy.spatial import cKDTree

    tree = cKDTree(pos.detach().cpu().numpy())
    pairs = tree.query_pairs(cutoff)
    if len(pairs) == 0:
        return torch.empty(2, 0, dtype=torch.long)
    edge_index = torch.tensor(list(pairs), dtype=torch.long).t()
    return to_undirected(edge_index, num_nodes=pos.shape[0])


def _edge_attr(pos, edge_index):
    src, dst = edge_index[0], edge_index[1]
    vec = pos[dst] - pos[src]
    dist = vec.norm(dim=-1)
    rbf = _rbf(dist)
    return rbf, vec.unsqueeze(-2)


class CATHDatasetS3FExact(Dataset):
    """CATH dataset backed by precomputed dMaSIF point clouds for the
    S3F-exact encoder.

    __getitem__ returns a torch_geometric Data with:
      - graph.x: placeholder ones (overwritten by ESM in model forward)
      - graph.node_pos: (n_res, 3) Cα
      - graph.edge_index, graph.edge_rbf, graph.edge_vec: precomputed
      - surface.verts, surface.vnormals, surface.x: (M, *) precomputed feats
      - surface.res2surf: (n_res, 60) fusion-pool indices
      - surface.edge_index, surface.edge_rbf, surface.edge_vec
      - sequence: str
      - masked_positions, mask_types, target_residues, random_aa_indices
    """

    SPLIT_RATIOS = (0.97, 0.02, 0.01)

    def __init__(
        self,
        precompute_dir: str,
        split: str,
        mask_rate: float = 0.15,
        max_length: int = 250,
        seed: int = 0,
    ):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split}")
        self.precompute_dir = precompute_dir
        self.split = split
        self.mask_rate = mask_rate
        self.max_length = max_length

        all_pts = sorted(f for f in os.listdir(precompute_dir) if f.endswith(".pt"))
        n = len(all_pts)
        n_train = int(n * self.SPLIT_RATIOS[0])
        n_val = int(n * self.SPLIT_RATIOS[1])
        if split == "train":
            self.files = all_pts[:n_train]
        elif split == "val":
            self.files = all_pts[n_train : n_train + n_val]
        else:
            self.files = all_pts[n_train + n_val :]

        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Data]:
        path = os.path.join(self.precompute_dir, self.files[idx])
        try:
            d = torch.load(path, weights_only=False, map_location="cpu")
        except Exception as e:
            logger.warning("Load failed for %s: %s", path, e)
            return None

        sequence = d["sequence"]
        ca_pos = d["ca_pos"].float()
        bb_pos = d["bb_pos"].float()
        surf_pos = d["surf_pos"].float()
        surf_normals = d["surf_normals"].float()
        surf_feat = d["surf_feat"].float()

        n_res = len(sequence)
        n_surf = surf_pos.shape[0]
        if n_res < 32 or n_surf < 64:
            return None

        crop_window = None
        if self.max_length is not None and n_res > self.max_length:
            start = int(self._rng.integers(0, n_res - self.max_length + 1))
            crop_window = (start, start + self.max_length)
            s, e = crop_window
            sequence = sequence[s:e]
            ca_pos = ca_pos[s:e]
            bb_pos = bb_pos[s:e]
            n_res = e - s

            surf_to_res = torch.cdist(surf_pos, ca_pos).argmin(dim=1)
            keep = (surf_to_res >= 0) & (surf_to_res < n_res)
            surf_pos = surf_pos[keep]
            surf_normals = surf_normals[keep]
            surf_feat = surf_feat[keep]
            n_surf = surf_pos.shape[0]
            if n_surf < 64:
                return None

        res_edges = _radius_edges(ca_pos, RADIUS_CUTOFF)
        res_rbf, res_vec = _edge_attr(ca_pos, res_edges)

        surf_edges = knn_graph(surf_pos, k=SURF_KNN, loop=False)
        surf_rbf, surf_vec = _edge_attr(surf_pos, surf_edges)

        res2surf = self._compute_res2surf(bb_pos, surf_pos)

        masked_positions = self._sample_positions(n_res)
        mask_types, target_residues, random_aa_indices, valid = self._masking_plan(
            sequence, masked_positions
        )
        masked_positions = masked_positions[valid]
        mask_types = mask_types[valid]
        target_residues = target_residues[valid]
        random_aa_indices = random_aa_indices[valid]
        if len(masked_positions) == 0:
            return None

        graph = Data(
            x=torch.ones(n_res, 1),
            node_pos=ca_pos,
            edge_index=res_edges,
            edge_rbf=res_rbf,
            edge_vec=res_vec,
        )
        surface = S3FSurfaceData(
            verts=surf_pos,
            vnormals=surf_normals,
            x=surf_feat,
            pos=surf_pos,
            edge_index=surf_edges,
            edge_rbf=surf_rbf,
            edge_vec=surf_vec,
            res2surf=res2surf,
        )

        return Data(
            graph=graph,
            surface=surface,
            sequence=sequence,
            masked_positions=masked_positions,
            mask_types=mask_types,
            target_residues=target_residues,
            random_aa_indices=random_aa_indices,
            protein_name=self.files[idx].replace(".pt", ""),
        )

    @staticmethod
    def _compute_res2surf(bb_pos, surf_pos):
        """For each backbone atom (n_res*3,), find 20 NN surface points.
        Returns (n_res, 60) flattened index tensor."""
        n_res = bb_pos.shape[0]
        bb_flat = bb_pos.reshape(-1, 3)
        dists = torch.cdist(bb_flat, surf_pos)
        k = min(FUSION_K_PER_ATOM, surf_pos.shape[0])
        _, nn_idx = dists.topk(k, dim=1, largest=False)
        return nn_idx.view(n_res, 3 * k)

    @staticmethod
    def _compute_surf2res(surf_pos, ca_pos, k):
        dists = torch.cdist(surf_pos, ca_pos)
        k_eff = min(k, ca_pos.shape[0])
        _, nn_idx = dists.topk(k_eff, dim=1, largest=False)
        return nn_idx

    def _sample_positions(self, n_res: int) -> torch.Tensor:
        n_mask = max(1, int(round(n_res * self.mask_rate)))
        n_mask = min(n_mask, n_res)
        positions = self._rng.choice(n_res, size=n_mask, replace=False)
        return torch.from_numpy(positions).long()

    def _masking_plan(self, sequence, positions):
        LETTER_TO_IDX = {
            "A": 0,
            "R": 1,
            "N": 2,
            "D": 3,
            "C": 4,
            "Q": 5,
            "E": 6,
            "G": 7,
            "H": 8,
            "I": 9,
            "L": 10,
            "K": 11,
            "M": 12,
            "F": 13,
            "P": 14,
            "S": 15,
            "T": 16,
            "W": 17,
            "Y": 18,
            "V": 19,
        }
        n_mask = len(positions)
        r = self._rng.random(n_mask)
        mask_types = np.where(
            r < 0.8,
            MASK_TYPE_MASK,
            np.where(r < 0.9, MASK_TYPE_RANDOM, MASK_TYPE_UNCHANGED),
        )

        target_residues = np.full(n_mask, -1, dtype=np.int64)
        for i, pos in enumerate(positions):
            letter = sequence[int(pos)]
            target_residues[i] = LETTER_TO_IDX.get(letter, -1)

        valid = target_residues >= 0
        random_aa = self._rng.choice(20, size=n_mask)
        random_aa_indices = np.where(mask_types == MASK_TYPE_RANDOM, random_aa, -1)

        return (
            torch.from_numpy(mask_types).long(),
            torch.from_numpy(target_residues).long(),
            torch.from_numpy(random_aa_indices).long(),
            torch.from_numpy(valid).bool(),
        )
