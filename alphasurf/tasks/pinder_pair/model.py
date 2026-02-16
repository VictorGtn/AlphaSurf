"""
PINDER-Pair network: Protein pair encoder + interaction classifier.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
from alphasurf.networks.protein_encoder import ProteinEncoder


def _offset_and_concat(
    indices: Union[List[torch.Tensor], torch.Tensor],
    cum_lengths: torch.Tensor,
    split_labels: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Apply cumulative offsets to batched indices and concatenate.

    Args:
        indices: Per-sample index tensors (list) or flat concatenated tensor.
        cum_lengths: Cumulative node counts from torch.cumsum(lengths, dim=0).
        split_labels: If indices is a flat tensor, use label list lengths to infer splits.

    Returns:
        Flat tensor of offset-corrected indices.
    """
    if isinstance(indices, list):
        # Clone to avoid mutating the batch
        parts = [indices[0].clone()]
        for i in range(1, len(indices)):
            parts.append(indices[i] + cum_lengths[i - 1])
        return torch.cat(parts)
    else:
        # Tensor format
        if split_labels is not None and isinstance(split_labels, list):
            parts = []
            start = 0
            for i, lbl in enumerate(split_labels):
                n = len(lbl)
                chunk = indices[start : start + n]
                if i > 0:
                    chunk = chunk + cum_lengths[i - 1]
                parts.append(chunk)
                start += n
            return torch.cat(parts)
        else:
            return indices.reshape(-1)


class PinderPairNet(nn.Module):
    """
    Network for protein-protein interface prediction.

    Architecture:
    1. Shared ProteinEncoder processes both proteins
    2. Extract features at interface residue positions
    3. Predict binding site probability for each residue
    4. Return embeddings for metric learning
    """

    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.encoder = ProteinEncoder(cfg_encoder)

        encoded_dims = cfg_head.encoded_dims

        # Binding Site Prediction Head
        self.binding_site_head = nn.Sequential(
            nn.Linear(encoded_dims, encoded_dims),
            nn.ReLU(),
            nn.Dropout(p=cfg_head.dropout),
            nn.Linear(encoded_dims, 1),
        )

        # We use the raw embeddings for metric learning,
        # but we could add a projection head here if needed.
        # For now, following the plan to use encoder outputs.

    def forward(self, batch):
        # Encode both proteins with shared encoder
        surface_1, graph_1 = self.encoder(graph=batch.graph_1, surface=batch.surface_1)
        surface_2, graph_2 = self.encoder(graph=batch.graph_2, surface=batch.surface_2)

        # --- Graph / Residue Level ---

        # 1. Binding Site Predictions (for all nodes)
        site_pred_1 = self.binding_site_head(graph_1.x)
        site_pred_2 = self.binding_site_head(graph_2.x)

        # 2. Extract features at pair positions
        # Apply cumulative offsets to batched graph indices
        base_left = torch.cumsum(batch.g1_len, dim=0)
        base_right = torch.cumsum(batch.g2_len, dim=0)

        idx_left = _offset_and_concat(batch.idx_left, base_left, batch.label)
        idx_right = _offset_and_concat(batch.idx_right, base_right, batch.label)

        emb_left = graph_1.x[idx_left]
        emb_right = graph_2.x[idx_right]

        # Use indexing to get site predictions for the pairs (if needed for debugging)
        # site_pred_left = site_pred_1[idx_left]
        # site_pred_right = site_pred_2[idx_right]

        # --- Surface Level ---

        surf_out = None
        if (
            hasattr(batch, "surface_idx_left")
            and batch.surface_idx_left is not None
            and len(batch.surface_idx_left) > 0
        ):
            # 1. Surface Site Predictions
            surf_site_pred_1 = self.binding_site_head(surface_1.x)
            surf_site_pred_2 = self.binding_site_head(surface_2.x)

            # 2. Extract features at pair positions
            s1_lens = torch.tensor(
                [s.x.shape[0] for s in batch.surface_1.to_data_list()],
                device=batch.surface_1.x.device,
            )
            s2_lens = torch.tensor(
                [s.x.shape[0] for s in batch.surface_2.to_data_list()],
                device=batch.surface_2.x.device,
            )

            base_l_surf = torch.cumsum(s1_lens, dim=0)
            base_r_surf = torch.cumsum(s2_lens, dim=0)

            surf_idx_l = _offset_and_concat(
                batch.surface_idx_left, base_l_surf, batch.surface_label
            )
            surf_idx_r = _offset_and_concat(
                batch.surface_idx_right, base_r_surf, batch.surface_label
            )

            processed_left_surf = surface_1.x[surf_idx_l]
            processed_right_surf = surface_2.x[surf_idx_r]

        # Prepare output dictionary
        outputs = {}

        # Graph outputs
        outputs["graph"] = {
            "emb_left": emb_left,
            "emb_right": emb_right,
            "site_pred_1": site_pred_1,
            "site_pred_2": site_pred_2,
        }

        # Surface outputs (if available)
        if hasattr(batch, "surface_1"):
            outputs["surface"] = {
                "emb_left": processed_left_surf,
                "emb_right": processed_right_surf,
                "site_pred_1": surf_site_pred_1,
                "site_pred_2": surf_site_pred_2,
            }
        else:
            outputs["surface"] = None

        return outputs
