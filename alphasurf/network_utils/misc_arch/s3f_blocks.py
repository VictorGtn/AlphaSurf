"""
Composable S3F building blocks for the ProteinEncoder system.

Each block has signature forward(surface, graph) -> (surface, graph),
matching ProteinEncoderBlock. The S3F dual-branch GVP architecture is
assembled by stacking these blocks in the encoder YAML:

  1. S3FStructureInit  — lift ESM (1280) -> GVP (256,16) on the Cα radius graph
  2. S3FSurfaceInit    — init surface from batch-safe 3-NN residue ESM + curvature
  3. S3FStructureGVP   — N GVPConvLayers on the residue graph (structure branch)
  4. S3FSurfaceGVP     — N GVPConvLayers on the surface point cloud (surface branch)
  5. S3FFusion         — pool surface -> residues, add additively, write graph.x

Branches 3 and 4 run independently (no cross-talk). The fusion block
combines their outputs into graph.x for the downstream residue head.

Intermediate GVP state is carried on the graph/surface objects as
.gvp_s, .gvp_v (node tuples) and .gvp_edge_* (edge tuples + topology).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from alphasurf.network_utils.misc_arch.gvp_gnn import (
    GVP,
    GVPConvLayer,
    LayerNorm as GVPLayerNorm,
)
from alphasurf.network_utils.communication.passing_utils import _rbf

ESM_DIM = 1280
RBF_DIM = 16
RADIUS_CUTOFF = 10.0
RBF_D_MAX = 20.0
HKS_DIM = 32
CURV_DIM = 10
SURF_FEAT_DIM = HKS_DIM + CURV_DIM


def _surface_residue_knn(
    res_pos, surf_pos, k, res_batch=None, surf_batch=None
):
    """Batch-safe residue neighbours and Euclidean distances per surface point."""
    from torch_geometric.nn import knn

    if res_batch is None:
        res_batch = torch.zeros(len(res_pos), dtype=torch.long, device=res_pos.device)
    if surf_batch is None:
        surf_batch = torch.zeros(
            len(surf_pos), dtype=torch.long, device=surf_pos.device
        )

    assignment = knn(
        x=res_pos,
        y=surf_pos,
        k=k,
        batch_x=res_batch,
        batch_y=surf_batch,
    )
    surf_idx, res_idx = assignment

    # torch-cluster groups the result by query node. Sort explicitly so the
    # reshape remains correct across supported PyG/torch-cluster versions.
    order = torch.argsort(surf_idx, stable=True)
    surf_idx = surf_idx[order]
    res_idx = res_idx[order]
    expected_surf_idx = torch.arange(
        len(surf_pos), device=surf_pos.device
    ).repeat_interleave(k)
    if not torch.equal(surf_idx, expected_surf_idx):
        raise RuntimeError("S3F surface-to-residue kNN returned incomplete groups")

    nn_idx = res_idx.view(len(surf_pos), k)
    nn_dists = (surf_pos[:, None, :] - res_pos[nn_idx]).norm(dim=-1)
    return nn_idx, nn_dists


def _edge_features(pos, edge_index):
    """RBF (16-dim scalar, D_max=20, σ=1.25) + edge vector (1) per edge.

    D_max is decoupled from the radius-graph cutoff (S3F uses 20 Å basis
    span even though edges are <= 10 Å).
    """
    src, dst = edge_index[0], edge_index[1]
    vec = pos[dst] - pos[src]
    dist = vec.norm(dim=-1)
    rbf = _rbf(dist, D_min=0.0, D_max=RBF_D_MAX, D_count=RBF_DIM)
    return (rbf, vec.unsqueeze(-2))


class S3FStructureInit(nn.Module):
    """Lift ESM features to GVP dims on the precomputed Cα radius graph.

    Reads graph.x (ESM embeddings, [n_res, 1280]).
    Writes graph.gvp_s, graph.gvp_v (node tuple) and edge attributes.
    """

    def __init__(
        self,
        node_h_dim=(256, 16),
        edge_h_dim=(64, 1),
        vector_gate=True,
    ):
        super().__init__()
        self.residue_embedding = nn.Linear(ESM_DIM, ESM_DIM, bias=False)
        self.W_v = nn.Sequential(
            GVPLayerNorm((ESM_DIM, 0)),
            GVP((ESM_DIM, 0), node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            GVPLayerNorm((RBF_DIM, 1)), GVP((RBF_DIM, 1), edge_h_dim)
        )

    def forward(self, surface, graph):
        esm = graph.x
        h_node = self.residue_embedding(esm)
        s, v = self.W_v(h_node)

        # These radius edges were built independently per protein by the
        # dataset. PyG offsets them during batching, so reusing them prevents
        # accidental cross-protein edges.
        edge_index = graph.edge_index
        edge_s, edge_v = _edge_features(graph.node_pos, edge_index)
        edge_s, edge_v = self.W_e((edge_s, edge_v))

        graph.gvp_s = s
        graph.gvp_v = v
        graph.gvp_edge_index = edge_index
        graph.gvp_edge_s = edge_s
        graph.gvp_edge_v = edge_v
        return surface, graph


class S3FSurfaceInit(nn.Module):
    """Initialize surface GVP features from k-NN residues + precomputed geom feats.

    Matches S3F paper Eq. (4):
      h̃_i^(0) = MLP( f_i, (1/k) Σ_j Linear(h_j^(0), ||x̃_i − x_j||) )

    where f_i is the 42-dim precomputed geometric feature stored on surface.x
    (32-d HKS + 10-d multi-scale curvature from the precompute script). The
    inner Linear is bias-free over [ESM(1280) + 1].

    Reads graph.gvp_s (ESM-derived residue features) and surface.x.
    Writes surface.gvp_s, surface.gvp_v and surface edge attributes.
    """

    def __init__(
        self,
        node_h_dim=(256, 16),
        edge_h_dim=(64, 1),
        num_surf_res_neighbor=3,
        num_surf_graph_neighbor=16,
        surf_geom_dim=SURF_FEAT_DIM,
        vector_gate=True,
    ):
        super().__init__()
        self.k_res = num_surf_res_neighbor
        self.k_surf = num_surf_graph_neighbor

        self.surf_in_linear = nn.Linear(ESM_DIM + 1, ESM_DIM, bias=False)
        self.surf_in_mlp = nn.Sequential(
            nn.Linear(ESM_DIM + surf_geom_dim, 2 * ESM_DIM),
            nn.Dropout(0.1),
            nn.LayerNorm(2 * ESM_DIM),
            nn.ReLU(),
            nn.Linear(2 * ESM_DIM, ESM_DIM),
        )
        self.W_v = nn.Sequential(
            GVPLayerNorm((ESM_DIM, 0)),
            GVP((ESM_DIM, 0), node_h_dim, activations=(None, None)),
        )
        self.W_e = nn.Sequential(
            GVPLayerNorm((RBF_DIM, 1)), GVP((RBF_DIM, 1), edge_h_dim)
        )

    def forward(self, surface, graph):
        res_pos = graph.node_pos
        res_feat = graph.x.float()
        if res_feat.shape[-1] != ESM_DIM:
            raise RuntimeError(
                f"S3FSurfaceInit expects graph.x to be raw ESM ({ESM_DIM}-d); "
                f"got {res_feat.shape[-1]}. The model forward must put ESM into "
                f"graph.x before the encoder runs."
            )
        surf_pos = surface.verts.float()

        k = self.k_res
        nn_idx, nn_dists = _surface_residue_knn(
            res_pos,
            surf_pos,
            k,
            res_batch=getattr(graph, "batch", None),
            surf_batch=getattr(surface, "batch", None),
        )

        nn_feat = res_feat[nn_idx]
        nn_dists = nn_dists.unsqueeze(-1)

        per_neighbor = torch.cat([nn_feat, nn_dists], dim=-1)
        per_neighbor = self.surf_in_linear(per_neighbor)
        pooled = per_neighbor.mean(dim=1)

        surf_geom = surface.x.float()
        if surf_geom.shape[0] != surf_pos.shape[0]:
            surf_geom = surf_geom[: surf_pos.shape[0]]

        h_surf = self.surf_in_mlp(torch.cat([pooled, surf_geom], dim=-1))

        s, v = self.W_v(h_surf)

        # As for the residue graph, surface kNN edges are constructed per
        # protein by the dataset and made batch-safe by PyG.
        edge_index = surface.edge_index
        edge_s, edge_v = _edge_features(surf_pos, edge_index)
        edge_s, edge_v = self.W_e((edge_s, edge_v))

        surface.gvp_s = s
        surface.gvp_v = v
        surface.gvp_edge_index = edge_index
        surface.gvp_edge_s = edge_s
        surface.gvp_edge_v = edge_v
        return surface, graph


class S3FStructureGVP(nn.Module):
    """Run N GVPConvLayers on the residue graph (structure branch)."""

    def __init__(
        self,
        node_h_dim=(256, 16),
        edge_h_dim=(64, 1),
        num_layers=5,
        drop_rate=0.1,
        vector_gate=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    node_h_dim, edge_h_dim, drop_rate=drop_rate, vector_gate=vector_gate
                )
                for _ in range(num_layers)
            ]
        )
        self.W_out = nn.Sequential(
            GVPLayerNorm(node_h_dim),
            GVP(node_h_dim, (node_h_dim[0], 0), activations=(F.relu, None)),
        )

    def forward(self, surface, graph):
        h = (graph.gvp_s, graph.gvp_v)
        edge_attr = (graph.gvp_edge_s, graph.gvp_edge_v)
        for layer in self.layers:
            h = layer(h, graph.gvp_edge_index, edge_attr)
        out = self.W_out(h)
        graph.gvp_out = out[0]
        return surface, graph


class S3FSurfaceGVP(nn.Module):
    """Run N GVPConvLayers on the surface point cloud (surface branch)."""

    def __init__(
        self,
        node_h_dim=(256, 16),
        edge_h_dim=(64, 1),
        num_layers=5,
        drop_rate=0.1,
        vector_gate=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    node_h_dim, edge_h_dim, drop_rate=drop_rate, vector_gate=vector_gate
                )
                for _ in range(num_layers)
            ]
        )
        self.W_out = nn.Sequential(
            GVPLayerNorm(node_h_dim),
            GVP(node_h_dim, (node_h_dim[0], 0), activations=(F.relu, None)),
        )

    def forward(self, surface, graph):
        h = (surface.gvp_s, surface.gvp_v)
        edge_attr = (surface.gvp_edge_s, surface.gvp_edge_v)
        for layer in self.layers:
            h = layer(h, surface.gvp_edge_index, edge_attr)
        out = self.W_out(h)
        surface.gvp_out = out[0]
        return surface, graph


class S3FFusion(nn.Module):
    """Pool surface features to residues, fuse additively with structure output.

    Matches S3F released code: for each backbone atom (N/CA/C), find k=20
    nearest surface points; mean-pool over all 60 per residue. Uses the
    precomputed `surface.res2surf` (n_res, 60) index from the precompute
    script; S3FSurfaceData's __inc__ makes the indices batch-safe.

    Reads graph.gvp_out (structure) and surface.gvp_out (surface).
    Writes fused [n_res, dim] into graph.x.
    """

    def __init__(self, node_h_dim=(256, 16), num_surf_res_neighbor=20):
        super().__init__()
        self.k_pool = num_surf_res_neighbor

    def forward(self, surface, graph):
        bb_feat = graph.gvp_out
        surf_feat = surface.gvp_out

        res2surf = getattr(surface, "res2surf", None)
        if res2surf is None or res2surf.shape[0] != bb_feat.shape[0]:
            raise RuntimeError(
                "S3FFusion requires surface.res2surf (n_res, 60) — "
                "the precomputed 60-NN index. Run precompute_s3f_exact.py."
            )

        k_total = min(res2surf.shape[1], surf_feat.shape[0])
        idx = res2surf[:, :k_total]
        pooled = surf_feat[idx].mean(dim=1)

        graph.x = bb_feat + pooled
        return surface, graph
