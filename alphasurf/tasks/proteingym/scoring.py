"""
Embedding-delta scoring for ProteinGym (Option D in the task README).

For each mutant, the WT graph is cloned, residue-type one-hot features at
mutated positions are overwritten with the mutant amino acid, and the cloned
graph is re-encoded. Score = -sum over mutated positions of the L2 distance
between MT and WT per-residue embeddings. Lower distance (more similar to WT)
yields a higher (better) score, hence the sign flip.

This is a zero-shot, training-free heuristic, not a log-odds; reported
Spearman numbers are not directly comparable to ProteinGym leaderboard rows.
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
