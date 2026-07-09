"""
S3FPretrainNet: AlphaSurf encoder + frozen ESM2-650M + masked-residue head.

Mirrors S3F's pretraining architecture with AlphaSurf's CGAL alpha-complex
surface and ProNet+DiffusionNet+GVP fusion encoder replacing S3F's GVP-GNN.

Forward pass:
  1. Tokenize sequences, apply mask plan to ESM tokens
  2. Run frozen ESM2-650M -> per-residue 1280-dim embeddings
  3. Concatenate ESM embeddings into graph.x (31 -> 1311 dims)
  4. Apply mask plan to graph.x (AA one-hot + hphob)
  5. Zero surface features at leaky vertices (k=20 nearest to masked Ca)
  6. Run ProteinEncoder -> (N_res, 128) per-residue embeddings
  7. Residue head (Dropout + Linear) -> (N_res, 21) logits
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from alphasurf.networks.protein_encoder import ProteinEncoder
from alphasurf.protein.graphs import res_type_dict, res_type_to_hphob

logger = logging.getLogger(__name__)

ESM_EMBED_DIM = 1280
ESM_REPR_LAYER = 33
NUM_AA_CLASSES = 21

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


class S3FPretrainNet(nn.Module):
    def __init__(self, cfg_encoder, cfg_head):
        super().__init__()
        self.encoder = ProteinEncoder(cfg_encoder)
        self.encoded_dim = cfg_head.encoded_dims
        self.head_dropout = cfg_head.dropout

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

        x = torch.cat([graph.x, esm_emb], dim=-1)
        x = self._apply_node_mask(x, per_protein, ptr, B)

        graph.x = x
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
