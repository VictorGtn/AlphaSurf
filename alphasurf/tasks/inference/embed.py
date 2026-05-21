#!/usr/bin/env python3
"""
Embed proteins into graph and surface latent vectors.

Usage:
    # Single PDB
    python tasks/inference/embed.py --ckpt last.ckpt --pdb protein.pdb

    # Directory of PDBs (batched)
    python tasks/inference/embed.py --ckpt last.ckpt --pdb-dir /path/to/pdbs/
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule
from alphasurf.utils.config_utils import merge_surface_config
from alphasurf.utils.data_utils import AtomBatch


class ProteinEmbedDataset(Dataset):
    """Loads PDB files into protein objects (surface + graph) using ProteinLoader."""

    def __init__(self, pdb_paths, protein_loader):
        self.pdb_paths = pdb_paths
        self.loader = protein_loader

    def __len__(self):
        return len(self.pdb_paths)

    def __getitem__(self, idx):
        pdb_path = self.pdb_paths[idx]
        name = Path(pdb_path).stem
        # ProteinLoader handles: PDB -> alpha complex surface -> graph -> features
        protein = self.loader.load(name, pdb_path=pdb_path)
        if protein is None:
            return None
        return {"name": name, "surface": protein.surface, "graph": protein.graph}


def collate_fn(batch):
    """Batch proteins using AtomBatch (handles SurfaceObject, ResidueGraph, etc.)."""
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    names = [x["name"] for x in batch]
    atom_batch = AtomBatch.from_data_list(batch)
    return names, atom_batch


def load_model(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = PinderPairModule.load_from_checkpoint(ckpt_path, map_location=device)
    module.eval()
    module.to(device)
    return module, device


def build_protein_loader(cfg):
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Embed proteins into graph and surface latent vectors using a trained\n"
            "PINDER-Pair encoder. Generates alpha complex surfaces on the fly.\n"
            "\n"
            "Output .pt contents per protein:\n"
            "  graph_embedding   (N_residues, D) per-residue graph embeddings\n"
            "  surface_embedding (N_verts, D) per-vertex surface embeddings\n"
            "  graph_node_pos    (N_residues, 3) residue CA coordinates\n"
            "  surface_verts     (N_verts, 3) mesh vertex coordinates\n"
            "  name              protein name (from PDB filename stem)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to a trained PINDER-Pair model checkpoint (.ckpt). "
        "All config (encoder, surface method, etc.) is read from the checkpoint.",
    )
    pdb_group = parser.add_mutually_exclusive_group(required=True)
    pdb_group.add_argument(
        "--pdb",
        default=None,
        help="Path to a single PDB file to embed.",
    )
    pdb_group.add_argument(
        "--pdb-dir",
        default=None,
        help="Directory containing .pdb files. All files matching *.pdb will be embedded.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output .pt path for a single protein. Auto-named if omitted. "
        "Ignored when using --pdb-dir (use --output-dir instead).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output .pt files. Each protein gets {name}_embed.pt. "
        "Used with --pdb-dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of proteins per encoder forward pass. Default: 4.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes for parallel surface generation. "
        "0 = load in main process. Default: 0.",
    )
    args = parser.parse_args()

    # Collect PDB paths
    if args.pdb:
        pdb_paths = [args.pdb]
    else:
        pdb_paths = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdb_paths:
        print("Error: no PDB files found")
        sys.exit(1)

    module, device = load_model(args.ckpt)
    # Build on-the-fly loader: reads surface method, alpha value, etc. from checkpoint config
    loader = build_protein_loader(module.hparams.cfg)

    dataset = ProteinEmbedDataset(pdb_paths, loader)
    # DataLoader handles parallel loading (num_workers) and batching
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        shuffle=False,
    )

    total = len(pdb_paths)
    done = 0

    for names, atom_batch in dataloader:
        if atom_batch is None:
            continue

        # Extract batched surface and graph from AtomBatch
        surface = atom_batch.surface.to(device)
        graph = atom_batch.graph.to(device)

        # Run the shared encoder (all blocks before the prediction heads)
        with torch.no_grad():
            surface_emb, graph_emb = module.model.encoder(graph=graph, surface=surface)

        # Split embeddings back per-protein using PyG batch pointers
        s_ptr = surface_emb.ptr
        g_ptr = graph_emb.ptr

        for i, name in enumerate(names):
            g_start, g_end = g_ptr[i].item(), g_ptr[i + 1].item()
            s_start, s_end = s_ptr[i].item(), s_ptr[i + 1].item()

            result = {
                "graph_embedding": graph_emb.x[g_start:g_end].cpu(),
                "surface_embedding": surface_emb.x[s_start:s_end].cpu(),
                "graph_node_pos": atom_batch.graph.node_pos[g_start:g_end].cpu(),
                "surface_verts": atom_batch.surface.verts[s_start:s_end].cpu(),
                "name": name,
            }

            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                out_path = os.path.join(args.output_dir, f"{name}_embed.pt")
            elif args.output and total == 1:
                out_path = args.output
            else:
                out_path = f"{name}_embed.pt"

            torch.save(result, out_path)
            done += 1
            print(
                f"[{done}/{total}] {name}: graph {result['graph_embedding'].shape}, "
                f"surface {result['surface_embedding'].shape} -> {out_path}"
            )


if __name__ == "__main__":
    main()
