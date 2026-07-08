"""
Encoder loading and per-protein forward pass for ProteinGym scoring.

Option D: load the PINDER checkpoint (PinderPairModule), run the encoder
unchanged, score via embedding-delta on a cloned graph.
Option F: load the S3F checkpoint (S3FPretrainModule), run the full model
(encoder + ESM + residue head), score via log-odds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule
from alphasurf.utils.config_utils import merge_surface_config
from alphasurf.utils.data_utils import AtomBatch


def load_encoder_module(ckpt_path: str | Path) -> Tuple[PinderPairModule, str]:
    """Load the PINDER checkpoint and put the encoder in eval mode."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = PinderPairModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    module.eval()
    module.to(device)
    return module, device


def load_s3f_module(ckpt_path: str | Path):
    """Load the S3F checkpoint and put the model in eval mode."""
    from alphasurf.tasks.s3f_pretrain.pl_model import S3FPretrainModule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = S3FPretrainModule.load_from_checkpoint(str(ckpt_path), map_location=device)
    module.eval()
    module.to(device)
    return module, device


def build_protein_loader(module) -> ProteinLoader:
    """Rebuild the on-the-fly ProteinLoader from the checkpoint config."""
    cfg = module.hparams.cfg
    on_fly_cfg = getattr(cfg, "on_fly", None)
    surface_config = merge_surface_config(cfg.cfg_surface, on_fly_cfg)
    graph_config = merge_surface_config(cfg.cfg_graph, on_fly_cfg)
    surface_config.use_poisson = "poisson" in cfg.encoder.name
    return ProteinLoader(
        mode="on_fly",
        pdb_dir="",
        surface_config=surface_config,
        graph_config=graph_config,
    )


def encode_graphs(
    module: PinderPairModule,
    graphs,
    surfaces,
    device: str,
) -> torch.Tensor:
    """Encode a batch of (graph, surface) pairs and return per-residue graph
    embeddings concatenated along the node dimension with a `ptr` index.

    `graphs` and `surfaces` are lists of length B. Returns `(x, ptr)` where
    `x` has shape `(total_nodes, d)` and `ptr` has shape `(B + 1,)`.
    """
    data_list = [
        {"name": str(i), "surface": surfaces[i], "graph": graphs[i]}
        for i in range(len(graphs))
    ]
    atom_batch = AtomBatch.from_data_list(data_list)
    atom_batch.surface = atom_batch.surface.to(device)
    atom_batch.graph = atom_batch.graph.to(device)
    with torch.no_grad():
        _, graph_emb = module.model.encoder(
            graph=atom_batch.graph, surface=atom_batch.surface
        )
    return graph_emb.x, graph_emb.ptr


def encode_single_graph(module, graph, surface, device: str) -> torch.Tensor:
    """Encode one protein and return its `(N_residues, d)` graph embedding."""
    x, ptr = encode_graphs(module, [graph], [surface], device)
    return x[ptr[0].item() : ptr[1].item()]
