"""
Checkpoint loading and the on-the-fly ProteinLoader used by ProteinGym scoring.

Both scoring methods run the S3F checkpoint (S3FPretrainModule): `alphasurf`
uses the full model (encoder + ESM + residue head), `esm2` uses only its frozen
ESM-2 branch. `s3f_exact` checkpoints read an S3FReference instead of the
ProteinLoader output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.utils.config_utils import merge_surface_config


@dataclass
class S3FReference:
    """An AF2 structure in the form the s3f_exact encoder consumes."""

    sequence: str
    bb_pos: torch.Tensor  # (n_res, 3, 3) N/CA/C
    surface: dict  # full-protein surf_pos, surf_normals, surf_feat, res2surf
    b_factor: Optional[torch.Tensor]  # (n_res,) AF2 pLDDT


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


def load_s3f_reference(pdb_path, surface_dir) -> Optional[S3FReference]:
    """Load the precomputed S3F surface of an AF2 structure.

    `surface_dir` holds `<pdb file name>.pt` files in the precompute_s3f_exact
    format, written by precompute_alpha_s3f.py or convert_s3f_official_surfaces.py
    with the ProteinGym AF2 directory as input. The pLDDT is read from the PDB.
    """
    surface_path = Path(surface_dir) / f"{Path(pdb_path).name}.pt"
    if not surface_path.is_file():
        return None
    data = torch.load(surface_path, weights_only=False, map_location="cpu")
    surface = {
        key: data[key] for key in ("surf_pos", "surf_normals", "surf_feat", "res2surf")
    }

    b_factor = ProteinLoader._read_residue_b_factors(str(pdb_path))
    if b_factor is not None and len(b_factor) != len(data["sequence"]):
        b_factor = None
    return S3FReference(
        sequence=data["sequence"],
        bb_pos=data["bb_pos"].float(),
        surface=surface,
        b_factor=None if b_factor is None else torch.from_numpy(b_factor).float(),
    )
