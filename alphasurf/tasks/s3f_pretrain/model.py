"""
S3FPretrainNet: AlphaSurf encoder + frozen ESM2-650M + masked-residue head.

Mirrors S3F's pretraining architecture with AlphaSurf's CGAL alpha-complex
surface and ProNet+DiffusionNet+GVP fusion encoder replacing S3F's GVP-GNN.

Forward pass:
  1. Tokenize sequences, apply mask plan to ESM tokens
  2. Run frozen ESM2-650M -> per-residue 1280-dim embeddings
  3. Concatenate ESM embeddings into graph.x (31 -> 1311 dims)
  4. Apply mask plan to graph.x (AA one-hot + hphob)
  5. Optionally lift masked ESM + distance from 3 nearby residues to each
     AlphaSurf vertex and concatenate with its geometric features
  6. Run ProteinEncoder -> per-residue embeddings
  7. Residue head (Dropout + Linear) -> (N_res, 20) logits

For the AlphaSurf path, 3D masking happens before this forward pass by removing
side-chain atoms and regenerating the mesh. The s3f_exact point cloud is built
from N/CA/C only and therefore needs no point-level masking.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from alphasurf.networks.protein_encoder import ProteinEncoder
from alphasurf.network_utils.misc_arch.s3f_blocks import _surface_residue_knn
from alphasurf.protein.graphs import res_type_dict, res_type_to_hphob

logger = logging.getLogger(__name__)

ESM_EMBED_DIM = 1280
ESM_REPR_LAYER = 33
NUM_AA_CLASSES = 20
S3F_EXACT_OUTPUT_DIM = 256

RES_TYPE_TO_LETTER = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class SurfaceESMInjector(nn.Module):
    """Lift masked residue ESM embeddings directly onto surface vertices.

    For each surface vertex, the ``k`` nearest C-alpha residues are found
    independently within each protein in the batch. A bias-free linear map of
    ``[ESM, distance]`` is mean-pooled over those neighbours. The linear map is
    evaluated as two terms so raw 1280-dimensional ESM vectors are projected
    once per residue rather than materialized ``k`` times per surface vertex.
    """

    def __init__(self, output_dim: int, k: int = 3):
        super().__init__()
        if k < 1:
            raise ValueError(f"surface_esm.k must be positive, got {k}")
        self.k = k
        self.esm_projection = nn.Linear(ESM_EMBED_DIM, output_dim, bias=False)
        self.distance_projection = nn.Linear(1, output_dim, bias=False)

    def forward(self, surface, graph, esm_emb):
        if surface is None or not hasattr(surface, "x") or surface.x is None:
            return surface

        nn_idx, nn_dists = _surface_residue_knn(
            graph.node_pos.float(),
            surface.verts.float(),
            self.k,
            res_batch=getattr(graph, "batch", None),
            surf_batch=getattr(surface, "batch", None),
        )
        residue_features = self.esm_projection(esm_emb.float())
        distance_features = self.distance_projection(nn_dists.unsqueeze(-1))
        lifted = (residue_features[nn_idx] + distance_features).mean(dim=1)
        surface.x = torch.cat(
            [surface.x, lifted.to(dtype=surface.x.dtype)],
            dim=-1,
        )
        return surface


class S3FPretrainNet(nn.Module):
    def __init__(self, cfg_encoder, cfg_head, cfg_surface_esm=None):
        super().__init__()
        self.encoder = ProteinEncoder(cfg_encoder)
        self.encoder_name = getattr(cfg_encoder, "name", "")
        self.is_s3f_exact = "s3f_exact" in self.encoder_name
        self.encoded_dim = (
            S3F_EXACT_OUTPUT_DIM if self.is_s3f_exact else cfg_head.encoded_dims
        )
        self.head_dropout = cfg_head.dropout
        surface_esm_enabled = bool(
            cfg_surface_esm is not None
            and getattr(cfg_surface_esm, "enabled", False)
            and not self.is_s3f_exact
        )
        self.surface_esm_injector = (
            SurfaceESMInjector(
                output_dim=cfg_head.encoded_dims,
                k=int(getattr(cfg_surface_esm, "k", 3)),
            )
            if surface_esm_enabled
            else None
        )

        self.residue_head = nn.Sequential(
            nn.Dropout(self.head_dropout),
            nn.Linear(self.encoded_dim, NUM_AA_CLASSES),
        )

        self.esm_model = None
        self.esm_alphabet = None
        self.esm_batch_converter = None
        self.esm_mask_idx = None
        self._res_type_to_esm_tok = None
        self._hphob_lookup = None
        self._esm_loaded = False

    def _load_esm(self, device: str):
        if self._esm_loaded:
            return
        import esm

        logger.info("Loading frozen ESM2-650M...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)
        self.esm_model = model
        self.esm_alphabet = alphabet
        self.esm_batch_converter = alphabet.get_batch_converter()
        self.esm_mask_idx = alphabet.mask_idx

        res_idx_to_esm = torch.zeros(NUM_AA_CLASSES, dtype=torch.long)
        for res_type, idx in res_type_dict.items():
            if idx >= NUM_AA_CLASSES:
                continue
            letter = RES_TYPE_TO_LETTER.get(res_type.upper())
            res_idx_to_esm[idx] = (
                alphabet.get_idx(letter) if letter else alphabet.unk_idx
            )
        self._res_type_to_esm_tok = res_idx_to_esm.to(device)

        hphob = torch.tensor(
            [res_type_to_hphob[i] for i in range(NUM_AA_CLASSES)], dtype=torch.float
        )
        self._hphob_lookup = hphob.to(device)
        self._esm_loaded = True
        logger.info("ESM2-650M loaded (frozen, on %s).", device)

    def forward(self, batch, device: str):
        if not self._esm_loaded:
            self._load_esm(device)

        graph = batch.graph
        surface = batch.surface
        sequences = batch.sequence
        if isinstance(sequences, str):
            sequences = [sequences]

        B = len(sequences)
        ptr = graph.ptr

        per_protein = self._collect_per_protein_mask_plan(batch, ptr, B, device)

        esm_emb = self._run_esm_masked(sequences, per_protein, device, graph.x.dtype)

        if "s3f_exact" in self.encoder_name:
            graph.x = esm_emb
        else:
            x = torch.cat([graph.x, esm_emb], dim=-1)
            x = self._apply_node_mask(x, per_protein, ptr, B)
            graph.x = x
            if self.surface_esm_injector is not None:
                surface = self.surface_esm_injector(surface, graph, esm_emb)

        _, graph_out = self.encoder(graph=graph, surface=surface)
        logits = self.residue_head(graph_out.x)

        global_masked = torch.cat(
            [p["masked"] + ptr[i] for i, p in enumerate(per_protein)]
        )
        target_residues = torch.cat([p["targets"] for p in per_protein])

        return {
            "logits": logits,
            "global_masked": global_masked,
            "target_residues": target_residues,
        }

    def _collect_per_protein_mask_plan(self, batch, ptr, B, device):
        mp_list = batch.masked_positions
        mt_list = batch.mask_types
        tg_list = batch.target_residues
        ra_list = batch.random_aa_indices

        if isinstance(mp_list, torch.Tensor):
            mp_list = [mp_list]
        if isinstance(mt_list, torch.Tensor):
            mt_list = [mt_list]
        if isinstance(tg_list, torch.Tensor):
            tg_list = [tg_list]
        if isinstance(ra_list, torch.Tensor):
            ra_list = [ra_list]

        per_protein = []
        for i in range(B):
            mp = (
                mp_list[i].to(device)
                if i < len(mp_list)
                else torch.empty(0, dtype=torch.long, device=device)
            )
            mt = (
                mt_list[i].to(device)
                if i < len(mt_list)
                else torch.empty(0, dtype=torch.long, device=device)
            )
            tg = (
                tg_list[i].to(device)
                if i < len(tg_list)
                else torch.empty(0, dtype=torch.long, device=device)
            )
            ra = (
                ra_list[i].to(device)
                if i < len(ra_list)
                else torch.empty(0, dtype=torch.long, device=device)
            )
            per_protein.append(
                {"masked": mp, "types": mt, "targets": tg, "random_aa": ra}
            )
        return per_protein

    @torch.no_grad()
    def _run_esm_masked(self, sequences, per_protein, device, dtype):
        data = [(str(i), seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.esm_batch_converter(data)
        tokens = tokens.to(device)

        for i, plan in enumerate(per_protein):
            mp = plan["masked"]
            if mp.numel() == 0:
                continue
            mt = plan["types"]
            ra = plan["random_aa"]
            tok_pos = 1 + mp

            is_mask = mt == 0
            is_random = mt == 1
            if is_mask.any():
                tokens[i, tok_pos[is_mask]] = self.esm_mask_idx
            if is_random.any():
                tokens[i, tok_pos[is_random]] = self._res_type_to_esm_tok[ra[is_random]]

        results = self.esm_model(tokens, repr_layers=[ESM_REPR_LAYER])
        esm_emb = results["representations"][ESM_REPR_LAYER]

        parts = []
        for i, seq in enumerate(sequences):
            n = len(seq)
            parts.append(esm_emb[i, 1 : 1 + n, :])
        return torch.cat(parts, dim=0).to(dtype)

    def _apply_node_mask(self, x, per_protein, ptr, B):
        x = x.clone()
        for i, plan in enumerate(per_protein):
            mp = plan["masked"]
            if mp.numel() == 0:
                continue
            global_pos = mp + ptr[i]
            mt = plan["types"]
            ra = plan["random_aa"]

            is_mask = mt == 0
            is_random = mt == 1

            if is_mask.any():
                x[global_pos[is_mask], 0] = 0.0
                x[global_pos[is_mask], 1:22] = 0.0
            if is_random.any():
                gp_rand = global_pos[is_random]
                ra_rand = ra[is_random]
                x[gp_rand, 1:22] = 0.0
                x[gp_rand, 1 + ra_rand] = 1.0
                x[gp_rand, 0] = self._hphob_lookup[ra_rand]
        return x
