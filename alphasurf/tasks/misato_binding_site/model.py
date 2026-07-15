"""AlphaSurf residue classifier for MISATO ligand binding sites."""

import torch.nn as nn
from alphasurf.networks.protein_encoder import ProteinEncoder


class MisatoBindingSiteNet(nn.Module):
    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.encoder = ProteinEncoder(cfg_encoder)
        dim = cfg_head.encoded_dims
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(cfg_head.dropout),
            nn.Linear(dim, 2),
        )

    def forward(self, batch):
        surface, graph = self.encoder(surface=batch.surface, graph=batch.graph)
        return {"logits": self.head(graph.x), "graph": graph, "surface": surface}
