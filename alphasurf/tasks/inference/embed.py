#!/usr/bin/env python3
"""
Embed a single protein into graph and surface latent vectors.

Usage:
    python tasks/inference/embed.py --ckpt last.ckpt --pdb protein.pdb
"""

import argparse
import os
import sys
from pathlib import Path

import torch

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule
from torch_geometric.data import Batch


def load_model(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = PinderPairModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    module.to(device)
    return module, device


def build_protein_loader(cfg, data_dir=None):
    on_fly_cfg = getattr(cfg, "on_fly", None)
    if on_fly_cfg is None:
        from alphasurf.utils.config_utils import merge_surface_config

        surface_dir = os.path.join(cfg.cfg_surface.data_dir, cfg.cfg_surface.data_name)
        graph_dir = os.path.join(cfg.cfg_graph.data_dir, cfg.cfg_graph.data_name)
        surface_config = merge_surface_config(cfg.cfg_surface, None)
        graph_config = merge_surface_config(cfg.cfg_graph, None)
        surface_config.use_poisson = "poisson" in cfg.encoder.name
        return ProteinLoader(
            mode="disk",
            pdb_dir=data_dir or "",
            surface_dir=surface_dir,
            graph_dir=graph_dir,
            surface_config=surface_config,
            graph_config=graph_config,
        )
    else:
        from alphasurf.utils.config_utils import merge_surface_config

        surface_config = merge_surface_config(cfg.cfg_surface, on_fly_cfg)
        graph_config = merge_surface_config(cfg.cfg_graph, on_fly_cfg)
        surface_config.use_poisson = "poisson" in cfg.encoder.name
        return ProteinLoader(
            mode="on_fly",
            pdb_dir="",
            surface_config=surface_config,
            graph_config=graph_config,
        )


def main():
    parser = argparse.ArgumentParser(description="Embed a protein into latent vectors")
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--pdb", required=True, help="Path to input PDB file")
    parser.add_argument(
        "--data-dir", default=None, help="Root data directory (disk mode only)"
    )
    parser.add_argument("--output", "-o", default=None, help="Output .pt path")
    args = parser.parse_args()

    module, device = load_model(args.ckpt)
    loader = build_protein_loader(module.hparams.cfg, args.data_dir)

    name = Path(args.pdb).stem
    protein = loader.load(name, pdb_path=args.pdb)
    if protein is None:
        print(f"Failed to load {args.pdb}")
        sys.exit(1)

    surface = Batch.from_data_list([protein.surface]).to(device)
    graph = Batch.from_data_list([protein.graph]).to(device)

    with torch.no_grad():
        surface_emb, graph_emb = module.model.encoder(graph=graph, surface=surface)

    out_path = args.output or f"{name}_embed.pt"
    torch.save(
        {
            "graph_embedding": graph_emb.x.cpu(),
            "surface_embedding": surface_emb.x.cpu(),
            "graph_node_pos": protein.graph.node_pos.cpu(),
            "surface_verts": protein.surface.verts.cpu(),
            "name": name,
        },
        out_path,
    )
    print(f"Saved embeddings to {out_path}")
    print(f"  Graph:   {graph_emb.x.shape}")
    print(f"  Surface: {surface_emb.x.shape}")


if __name__ == "__main__":
    main()
