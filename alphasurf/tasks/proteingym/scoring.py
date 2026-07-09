"""
Scoring for ProteinGym.

Option D (embedding-delta heuristic): clone WT graph, overwrite residue-type
one-hot at mutated positions, re-encode, score = -sum ||emb_MT - emb_WT||.
Training-free, not a log-odds; not comparable to leaderboard rows.

Option F (S3F-style log-odds): mask mutation positions, run the S3F-pretrained
encoder + residue head, score = sum [log P(MT | masked) - log P(WT | masked)].
Requires an S3FPretrainModule checkpoint. Produces a proper log-odds that IS
comparable to leaderboard rows (S3F, ESM-2, etc.).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from alphasurf.protein.graphs import (
    protein_letters_1to3,
    res_type_dict,
    res_type_to_hphob,
)
from alphasurf.tasks.proteingym.dataset import DMSAssay
from alphasurf.tasks.proteingym.model import encode_graphs, encode_single_graph


# graph.x layout: col 0 = hphob, cols 1..21 = AA one-hot (21 classes, UNK at 20),
# cols 22..30 = SSE one-hot. See ResidueGraphBuilder.arrays_to_resgraph.
RES_TYPE_ONEHOT_SLICE = slice(1, 22)
HPHOB_COL = 0


def aa_one_letter_to_idx(aa: str) -> int:
    """1-letter AA -> res_type_dict index. Non-standard residues map to UNK."""
    three = protein_letters_1to3.get(aa.upper(), "Unk").upper()
    return res_type_dict.get(three, res_type_dict["UNK"])


def clone_graph_with_mutation(graph, positions: Sequence[int], mt_aas: Sequence[str]):
    """Deep-copy a WT graph and overwrite the residue-type (and hphob) features
    at the given positions to encode the mutant amino-acid identity."""
    new_graph = copy.deepcopy(graph)
    x = new_graph.x.clone()
    for pos, aa in zip(positions, mt_aas):
        idx = aa_one_letter_to_idx(aa)
        x[pos, RES_TYPE_ONEHOT_SLICE] = 0.0
        x[pos, RES_TYPE_ONEHOT_SLICE.start + idx] = 1.0
        x[pos, HPHOB_COL] = res_type_to_hphob[idx]
    new_graph.x = x
    return new_graph


def _batched(items: List, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def score_assay(
    module,
    wt_graph,
    wt_surface,
    assay: DMSAssay,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Score every mutant in an assay with the embedding-delta heuristic.

    Returns a float array of length `len(assay.mutants)` aligned with the
    mutant order in `assay`.
    """
    wt_emb = encode_single_graph(module, wt_graph, wt_surface, device)

    mt_graphs = [
        clone_graph_with_mutation(wt_graph, m.positions, m.mt_aas)
        for m in assay.mutants
    ]
    mt_surfaces = [wt_surface for _ in mt_graphs]

    scores: List[float] = []
    for graph_batch, surface_batch in zip(
        _batched(mt_graphs, batch_size), _batched(mt_surfaces, batch_size)
    ):
        emb_x, ptr = encode_graphs(module, graph_batch, surface_batch, device)
        for i, g in enumerate(graph_batch):
            start, end = ptr[i].item(), ptr[i + 1].item()
            mt_emb = emb_x[start:end]
            mutant = assay.mutants[len(scores)]
            delta = (mt_emb[mutant.positions] - wt_emb[mutant.positions]).norm(dim=-1)
            scores.append(-float(delta.sum().item()))
    return np.array(scores, dtype=np.float64)


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Spearman / Pearson / MAE / RMSE between targets and predictions."""
    if len(targets) < 2:
        return {
            "spearmanr": float("nan"),
            "pearsonr": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    rho, _ = spearmanr(predictions, targets)
    r, _ = pearsonr(predictions, targets)
    mae = float(np.mean(np.abs(predictions - targets)))
    rmse = float(np.sqrt(np.mean((predictions - targets) ** 2)))
    return {
        "spearmanr": float(rho),
        "pearsonr": float(r),
        "mae": mae,
        "rmse": rmse,
    }


# ── Option F: S3F-style log-odds scoring ──────────────────────────────


def score_assay_option_f(
    module,
    loader,
    pdb_path,
    protein_name,
    assay: DMSAssay,
    device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Score mutants with S3F-style log-odds from the residue head.

    For each mutant, mask the mutation positions, run the encoder + head,
    take log_softmax at masked positions, and compute:
        score = sum [log P(MT_AA) - log P(WT_AA)]

    Each mutant's mutation positions are Ala-stripped from the coordinates
    before loading (surface + graph built from stripped structure).

    Returns a float array of length len(assay.mutants).
    """
    import torch as _torch
    from torch_geometric.data import Data as _Data

    from alphasurf.utils.data_utils import AtomBatch as _AtomBatch
    from alphasurf.protein.graphs import res_type_idx_to_1

    model = module.model
    if not model._esm_loaded:
        model._load_esm(device)

    scores: List[float] = []
    for mutant in assay.mutants:
        positions = list(mutant.positions)
        wt_aas = mutant.wt_aas
        mt_aas = mutant.mt_aas

        protein = loader.load(
            protein_name,
            pdb_path=pdb_path,
            ala_strip_positions=positions,
        )
        if protein is None or protein.graph is None or protein.surface is None:
            scores.append(float("nan"))
            continue

        graph = protein.graph
        surface = protein.surface
        aa_idx = graph.x[:, RES_TYPE_ONEHOT_SLICE].argmax(dim=-1).cpu().long()
        sequence = "".join(res_type_idx_to_1[i] for i in aa_idx.numpy())

        masked_positions = _torch.tensor(positions, dtype=_torch.long)
        mask_types = _torch.zeros(len(positions), dtype=_torch.long)
        target_residues = aa_idx[masked_positions]
        random_aa_indices = _torch.full((len(positions),), -1, dtype=_torch.long)

        sample = _Data(
            graph=graph,
            surface=surface,
            sequence=sequence,
            masked_positions=masked_positions,
            mask_types=mask_types,
            target_residues=target_residues,
            random_aa_indices=random_aa_indices,
        )
        batch = _AtomBatch.from_data_list([sample])

        model.eval()
        with _torch.no_grad():
            out = model(batch, device)
            logits = out["logits"]
            global_masked = out["global_masked"]

            masked_logits = logits[global_masked]
            log_probs = _torch.log_softmax(masked_logits, dim=-1)

            wt_idx = _torch.tensor(
                [aa_one_letter_to_idx(a) for a in wt_aas], device=device
            )
            mt_idx = _torch.tensor(
                [aa_one_letter_to_idx(a) for a in mt_aas], device=device
            )

            pos_range = _torch.arange(len(positions), device=device)
            log_p_wt = log_probs[pos_range, wt_idx]
            log_p_mt = log_probs[pos_range, mt_idx]
            score = (log_p_mt - log_p_wt).sum().item()
            scores.append(float(score))

    return np.array(scores, dtype=np.float64)
