"""
MaSIF-Ligand network: Protein encoder + ligand pocket pooling + classifier.
"""

import torch
import torch.nn as nn

from alphasurf.networks.protein_encoder import ProteinEncoder


class MasifLigandNet(nn.Module):
    """
    Network for ligand type classification from protein binding sites.

    Architecture:
    1. ProteinEncoder processes surface and/or graph
    2. Pool encoded features around ligand coordinates (k=10 nearest)
    3. MLP classifier predicts ligand type (7 classes)
    """

    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.encoder = ProteinEncoder(cfg_encoder)

        self.classifier = nn.Sequential(
            nn.Linear(cfg_head.encoded_dims, cfg_head.encoded_dims),
            nn.Dropout(p=cfg_head.dropout),
            nn.BatchNorm1d(cfg_head.encoded_dims),
            nn.SiLU(),
            nn.Linear(cfg_head.encoded_dims, cfg_head.output_dims),
        )

    def pool_around_ligand(
        self,
        pos: torch.Tensor,
        features: torch.Tensor,
        lig_coords: torch.Tensor,
        k: int = 10,
    ) -> torch.Tensor:
        """
        Pool features from k nearest vertices/nodes to ligand atoms.

        Args:
            pos: (N, 3) vertex or node positions
            features: (N, D) encoded features
            lig_coords: (M, 3) ligand atom coordinates
            k: Number of nearest neighbors to pool

        Returns:
            (D,) mean-pooled feature vector
        """
        with torch.no_grad():
            dists = torch.cdist(pos, lig_coords.float())
            nearest_indices = torch.topk(-dists, k=k, dim=0).indices.unique()
        return features[nearest_indices].mean(dim=0)

    def forward(self, batch):
        surface, graph = self.encoder(graph=batch.graph, surface=batch.surface)

        # Use surface features if available, else graph
        use_surface = False
        if surface is not None:
            try:
                surface_list = surface.to_data_list()
                if len(surface_list) > 0:
                    first = surface_list[0]
                    if (
                        hasattr(first, "verts")
                        and hasattr(first, "x")
                        and first.x is not None
                    ):
                        use_surface = True
            except (AttributeError, TypeError):
                pass

        if use_surface:
            pos_and_feats = [(s.verts, s.x) for s in surface.to_data_list()]
        else:
            pos_and_feats = [(g.node_pos, g.x) for g in graph.to_data_list()]

        pocket_embeddings = []
        for (pos, feats), lig_coord in zip(pos_and_feats, batch.lig_coord):
            emb = self.pool_around_ligand(pos, feats, lig_coord)
            pocket_embeddings.append(emb)

        x = torch.stack(pocket_embeddings)
        return self.classifier(x)
