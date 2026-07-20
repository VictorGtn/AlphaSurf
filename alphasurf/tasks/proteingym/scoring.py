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
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from alphasurf.protein.graphs import (
    protein_letters_1to3,
    res_type_dict,
    res_type_to_hphob,
)
from alphasurf.tasks.proteingym.dataset import ASSAY_RESIDUE_RANGES, DMSAssay
from alphasurf.tasks.proteingym.model import encode_graphs, encode_single_graph

logger = logging.getLogger("alphasurf.proteingym.scoring")

# graph.x layout: col 0 = hphob, cols 1..21 = AA one-hot (21 classes, UNK at 20),
# cols 22..30 = SSE one-hot. See ResidueGraphBuilder.arrays_to_resgraph.
RES_TYPE_ONEHOT_SLICE = slice(1, 22)
HPHOB_COL = 0
ESM_MAX_RESIDUES = 1022


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


def get_optimal_window(
    mutation_position: int,
    sequence_length: int,
    model_window: int = ESM_MAX_RESIDUES,
) -> Tuple[int, int]:
    """Mirror S3F's released long-sequence window selection."""
    if sequence_length <= model_window:
        return 0, sequence_length
    half_window = model_window // 2
    if mutation_position < half_window:
        return 0, model_window
    if mutation_position >= sequence_length - half_window:
        return sequence_length - model_window, sequence_length
    return mutation_position - half_window, mutation_position + half_window


def _scoring_window(
    assay: DMSAssay | str, structure_length: int, positions: Sequence[int]
) -> Tuple[int, int]:
    assay_id = assay.assay_id if isinstance(assay, DMSAssay) else assay
    residue_range = ASSAY_RESIDUE_RANGES.get(assay_id)
    if residue_range is not None:
        start, end = residue_range
        if structure_length == end - start:
            return 0, structure_length
        return start, end
    return get_optimal_window(positions[0], structure_length)


class MaskedGeometryDataset(Dataset):
    """Build one masked AlphaSurf graph/surface per unique mutation-site set."""

    def __init__(
        self,
        protein_loader,
        pdb_path: str,
        protein_name: str,
        assay_id: str,
        structure_length: int,
        group_items,
    ):
        self.protein_loader = protein_loader
        self.pdb_path = pdb_path
        self.protein_name = protein_name
        self.assay_id = assay_id
        self.structure_length = structure_length
        self.group_items = group_items

    def __len__(self):
        return len(self.group_items)

    def __getitem__(self, index):
        import torch

        from alphasurf.protein.graphs import res_type_idx_to_1

        positions_key, mutant_indices = self.group_items[index]
        positions = list(positions_key)
        start, end = _scoring_window(self.assay_id, self.structure_length, positions)
        relative_positions = [position - start for position in positions]
        if any(
            position < 0 or position >= end - start for position in relative_positions
        ):
            return None

        crop_window = (
            None if start == 0 and end == self.structure_length else (start, end)
        )
        try:
            protein = self.protein_loader.load(
                (f"{self.protein_name}_{start}_{end}_{'_'.join(map(str, positions))}"),
                pdb_path=self.pdb_path,
                crop_window=crop_window,
                ala_strip_positions=relative_positions,
                ala_strip_keep_cb=False,
            )
        except Exception as error:
            logger.warning(
                "[%s] failed to generate masked geometry at %s: %s",
                self.assay_id,
                positions,
                error,
            )
            return None
        if protein is None or protein.graph is None or protein.surface is None:
            return None

        graph = protein.graph
        aa_idx = graph.x[:, RES_TYPE_ONEHOT_SLICE].argmax(dim=-1).cpu().long()
        sequence = "".join(res_type_idx_to_1[i] for i in aa_idx.numpy())
        masked_positions = torch.tensor(relative_positions, dtype=torch.long)
        sample = Data(
            graph=graph,
            surface=protein.surface,
            sequence=sequence,
            masked_positions=masked_positions,
            mask_types=torch.zeros(len(positions), dtype=torch.long),
            target_residues=aa_idx[masked_positions],
            random_aa_indices=torch.full((len(positions),), -1, dtype=torch.long),
        )
        return sample, mutant_indices, len(positions)


def collate_masked_geometry(items):
    """Keep worker-built samples as a list for AtomBatch collation on the GPU host."""
    return [item for item in items if item is not None]


def score_assay_option_f(
    module,
    loader,
    pdb_path,
    protein_name,
    assay: DMSAssay,
    device: str,
    batch_size: int = 8,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    structure_length: int | None = None,
    reference_protein=None,
) -> np.ndarray:
    """Score mutants with S3F-style log-odds from the residue head.

    For each mutant, mask the mutation positions, run the encoder + head,
    take log_softmax at masked positions, and compute:
        score = sum [log P(MT_AA) - log P(WT_AA)]

    Mutants sharing the same set of positions reuse one masked forward pass,
    as in S3F's released evaluator. For AlphaSurf, the masked residues are
    reduced to N/CA/C/O and both graph and surface are regenerated, matching
    AlphaSurf S3F pretraining. Long sequences use S3F's 1,022-residue window.

    Returns a float array of length len(assay.mutants).
    """
    import torch as _torch

    from alphasurf.utils.data_utils import AtomBatch as _AtomBatch

    model = module.model

    if structure_length is None:
        reference = reference_protein or loader.load(protein_name, pdb_path=pdb_path)
        if reference is None or reference.graph is None:
            return np.full(len(assay.mutants), np.nan, dtype=np.float64)
        structure_length = int(reference.graph.x.shape[0])

    groups: Dict[Tuple[int, ...], List[int]] = {}
    for mutant_index, mutant in enumerate(assay.mutants):
        groups.setdefault(tuple(mutant.positions), []).append(mutant_index)

    geometry_dataset = MaskedGeometryDataset(
        protein_loader=loader,
        pdb_path=str(pdb_path),
        protein_name=protein_name,
        assay_id=assay.assay_id,
        structure_length=structure_length,
        group_items=list(groups.items()),
    )
    dataloader_args = {
        "dataset": geometry_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_masked_geometry,
        "pin_memory": False,
    }
    if num_workers > 0:
        dataloader_args["prefetch_factor"] = prefetch_factor
        dataloader_args["persistent_workers"] = False
    geometry_loader = DataLoader(**dataloader_args)
    logger.info(
        "[%s] generating %d unique masked geometries with %d workers",
        assay.assay_id,
        len(geometry_dataset),
        num_workers,
    )

    scores = np.full(len(assay.mutants), np.nan, dtype=np.float64)
    for geometry_batch in geometry_loader:
        if not geometry_batch:
            continue
        samples = [item[0] for item in geometry_batch]
        metadata = [(item[1], item[2]) for item in geometry_batch]

        batch = _AtomBatch.from_data_list(samples)
        batch.graph = batch.graph.to(device)
        batch.surface = batch.surface.to(device)
        model.eval()
        with _torch.no_grad():
            if not model._esm_loaded:
                model._load_esm(device)
            out = model(batch, device)
            masked_logits = out["logits"][out["global_masked"]]
            cursor = 0
            for mutant_indices, num_positions in metadata:
                log_probs = _torch.log_softmax(
                    masked_logits[cursor : cursor + num_positions], dim=-1
                )
                cursor += num_positions
                pos_range = _torch.arange(num_positions, device=device)
                for mutant_index in mutant_indices:
                    mutant = assay.mutants[mutant_index]
                    wt_idx = _torch.tensor(
                        [aa_one_letter_to_idx(a) for a in mutant.wt_aas],
                        device=device,
                    )
                    mt_idx = _torch.tensor(
                        [aa_one_letter_to_idx(a) for a in mutant.mt_aas],
                        device=device,
                    )
                    scores[mutant_index] = (
                        (log_probs[pos_range, mt_idx] - log_probs[pos_range, wt_idx])
                        .sum()
                        .item()
                    )

    return scores
