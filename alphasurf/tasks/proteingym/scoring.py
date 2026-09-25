"""
Scoring for ProteinGym.

`alphasurf`: mask the mutation positions, run the S3F-pretrained encoder +
residue head, score = sum [log P(MT | masked) - log P(WT | masked)]. Requires an
S3FPretrainModule checkpoint. A proper log-odds, comparable to leaderboard rows
(S3F, ESM-2, etc.).

`esm2`: the same log-odds from the frozen ESM-2 branch alone, ignoring
structure. The pure-sequence baseline.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from tqdm import tqdm

from alphasurf.protein.graphs import protein_letters_1to3, res_type_dict
from alphasurf.tasks.proteingym.dataset import ASSAY_RESIDUE_RANGES, DMSAssay

logger = logging.getLogger("alphasurf.proteingym.scoring")

# graph.x layout: col 0 = hphob, cols 1..21 = AA one-hot (21 classes, UNK at 20),
# cols 22..30 = SSE one-hot. See ResidueGraphBuilder.arrays_to_resgraph.
RES_TYPE_ONEHOT_SLICE = slice(1, 22)
ESM_MAX_RESIDUES = 1022


def aa_one_letter_to_idx(aa: str) -> int:
    """1-letter AA -> res_type_dict index. Non-standard residues map to UNK."""
    three = protein_letters_1to3.get(aa.upper(), "Unk").upper()
    return res_type_dict.get(three, res_type_dict["UNK"])


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    """Spearman / Pearson / MAE / RMSE between targets and predictions.

    Mutants whose score could not be produced come back as NaN. scipy
    propagates a single NaN into the correlation, which would drop the whole
    assay from the aggregate, so they are excluded here and counted instead.
    """
    scored = np.isfinite(predictions) & np.isfinite(targets)
    counts = {"num_mutants": int(len(targets)), "num_scored": int(scored.sum())}
    targets, predictions = targets[scored], predictions[scored]
    if len(targets) < 2:
        return {
            "spearmanr": float("nan"),
            "pearsonr": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            **counts,
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
        **counts,
    }


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


def _sequence_scoring_window(
    assay: DMSAssay | str, sequence_length: int, positions: Sequence[int]
) -> Tuple[int, int]:
    assay_id = assay.assay_id if isinstance(assay, DMSAssay) else assay
    residue_range = ASSAY_RESIDUE_RANGES.get(assay_id)
    if residue_range is not None:
        start, end = residue_range
        if sequence_length == end - start:
            return 0, sequence_length
        return start, end
    return get_optimal_window(positions[0], sequence_length)


def _scoring_window(
    assay: DMSAssay | str,
    sequence_length: int,
    positions: Sequence[int],
    *,
    structure_length: int | None = None,
    structure_offset: int = 0,
) -> Tuple[int, int] | None:
    """Map S3F's sequence window into structure coordinates.

    S3F chooses the window from the target sequence and then truncates the
    structure using its metadata range. ``positions`` are structure-relative
    when ``structure_offset`` is nonzero.
    """
    sequence_positions = [position - structure_offset for position in positions]
    if any(position < 0 or position >= sequence_length for position in sequence_positions):
        return None
    sequence_start, sequence_end = _sequence_scoring_window(
        assay, sequence_length, sequence_positions
    )
    if structure_length is None:
        return sequence_start, sequence_end

    start = sequence_start + structure_offset
    end = sequence_end + structure_offset
    if start < 0 or end > structure_length:
        return None
    return start, end


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
        structure_mask_mode: str = "backbone",
        sequence_length: int | None = None,
        structure_offset: int = 0,
    ):
        self.protein_loader = protein_loader
        self.pdb_path = pdb_path
        self.protein_name = protein_name
        self.assay_id = assay_id
        self.structure_length = structure_length
        self.sequence_length = (
            structure_length if sequence_length is None else sequence_length
        )
        self.structure_offset = structure_offset
        self.group_items = group_items
        if structure_mask_mode not in {"alanine", "backbone"}:
            raise ValueError(
                "structure_mask_mode must be alanine or backbone, got "
                f"{structure_mask_mode}"
            )
        self.structure_mask_mode = structure_mask_mode

    def __len__(self):
        return len(self.group_items)

    def __getitem__(self, index):
        import torch

        from alphasurf.protein.graphs import res_type_idx_to_1

        positions_key, mutant_indices = self.group_items[index]
        positions = list(positions_key)
        window = _scoring_window(
            self.assay_id,
            self.sequence_length,
            positions,
            structure_length=self.structure_length,
            structure_offset=self.structure_offset,
        )
        if window is None:
            logger.warning(
                "[%s] sequence crop for %s is not covered by the structure",
                self.assay_id,
                positions,
            )
            return None
        start, end = window
        relative_positions = [position - start for position in positions]

        crop_window = (
            None if start == 0 and end == self.structure_length else (start, end)
        )
        try:
            protein = self.protein_loader.load(
                (f"{self.protein_name}_{start}_{end}_{'_'.join(map(str, positions))}"),
                pdb_path=self.pdb_path,
                crop_window=crop_window,
                ala_strip_positions=relative_positions,
                ala_strip_keep_cb=self.structure_mask_mode == "alanine",
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
            structure_positions=torch.tensor(positions, dtype=torch.long),
            mask_types=torch.zeros(len(positions), dtype=torch.long),
            target_residues=aa_idx[masked_positions],
            random_aa_indices=torch.full((len(positions),), -1, dtype=torch.long),
        )
        return sample, mutant_indices, len(positions)


class S3FGeometryDataset(Dataset):
    """Build one s3f_exact sample per unique mutation-site set.

    The surface is built once from the full backbone and carries no side-chain
    information, so masking changes only the sequence, as in S3F. Windows are
    cut with the same res2surf correspondence crop used in pretraining.
    """

    def __init__(
        self,
        reference,
        assay_id: str,
        structure_length: int,
        group_items,
        sequence_length: int | None = None,
        structure_offset: int = 0,
    ):
        self.reference = reference
        self.assay_id = assay_id
        self.structure_length = structure_length
        self.sequence_length = (
            structure_length if sequence_length is None else sequence_length
        )
        self.structure_offset = structure_offset
        self.group_items = group_items

    def __len__(self):
        return len(self.group_items)

    def __getitem__(self, index):
        import torch

        from alphasurf.tasks.s3f_pretrain.dataset_s3f_exact import (
            LETTER_TO_IDX,
            CATHDatasetS3FExact,
            _edge_attr,
            _radius_edges,
            build_surface_data,
        )

        positions_key, mutant_indices = self.group_items[index]
        positions = list(positions_key)
        window = _scoring_window(
            self.assay_id,
            self.sequence_length,
            positions,
            structure_length=self.structure_length,
            structure_offset=self.structure_offset,
        )
        if window is None:
            logger.warning(
                "[%s] sequence crop for %s is not covered by the structure",
                self.assay_id,
                positions,
            )
            return None
        start, end = window
        relative_positions = [position - start for position in positions]

        surf = self.reference.surface
        if start == 0 and end == self.structure_length:
            surf_pos, surf_normals, surf_feat = (
                surf["surf_pos"],
                surf["surf_normals"],
                surf["surf_feat"],
            )
            res2surf = surf["res2surf"].reshape(end, -1)
        else:
            surf_pos, surf_normals, surf_feat, res2surf = (
                CATHDatasetS3FExact._crop_surface(
                    surf["surf_pos"],
                    surf["surf_normals"],
                    surf["surf_feat"],
                    surf["res2surf"][start:end],
                )
            )

        ca_pos = self.reference.bb_pos[start:end, 1].contiguous()
        edge_index = _radius_edges(ca_pos)
        edge_rbf, edge_vec = _edge_attr(ca_pos, edge_index)
        graph = Data(
            x=torch.ones(end - start, 1),
            node_pos=ca_pos,
            edge_index=edge_index,
            edge_rbf=edge_rbf,
            edge_vec=edge_vec,
        )
        sequence = self.reference.sequence[start:end]
        sample = Data(
            graph=graph,
            surface=build_surface_data(surf_pos, surf_normals, surf_feat, res2surf),
            sequence=sequence,
            masked_positions=torch.tensor(relative_positions, dtype=torch.long),
            structure_positions=torch.tensor(positions, dtype=torch.long),
            mask_types=torch.zeros(len(positions), dtype=torch.long),
            target_residues=torch.tensor(
                [LETTER_TO_IDX.get(sequence[p], -1) for p in relative_positions],
                dtype=torch.long,
            ),
            random_aa_indices=torch.full((len(positions),), -1, dtype=torch.long),
        )
        return sample, mutant_indices, len(positions)


def s3f_head_to_res_type_index():
    """Column index that reorders s3f_exact head logits into res_type_dict order.

    The s3f_exact head is trained on LETTER_TO_IDX classes, while scoring and
    the ESM logits use res_type_dict indices.
    """
    import torch

    from alphasurf.tasks.s3f_pretrain.dataset_s3f_exact import LETTER_TO_IDX

    index = torch.empty(len(LETTER_TO_IDX), dtype=torch.long)
    for letter, head_idx in LETTER_TO_IDX.items():
        index[aa_one_letter_to_idx(letter)] = head_idx
    return index


def collate_masked_geometry(items):
    """Keep worker-built samples as a list for AtomBatch collation on the GPU host."""
    return [item for item in items if item is not None], len(items)


def _score_esm_fallback_groups(
    model,
    assay: DMSAssay,
    groups,
    scores: np.ndarray,
    device: str,
    batch_size: int,
    sequence_length: int,
    structure_offset: int,
) -> int:
    """Fill failed structural groups with masked ESM-2 log-odds."""
    import torch as _torch

    if not model._esm_loaded:
        model._load_esm(device)

    fallback_groups = []
    for positions, mutant_indices in groups:
        sequence_positions = [position - structure_offset for position in positions]
        if any(
            position < 0 or position >= sequence_length
            for position in sequence_positions
        ):
            logger.warning(
                "[%s] cannot use ESM fallback for positions %s outside the sequence",
                assay.assay_id,
                positions,
            )
            continue
        sequence_start, sequence_end = _sequence_scoring_window(
            assay, sequence_length, sequence_positions
        )
        relative_positions = [position - sequence_start for position in sequence_positions]
        sequence = assay.wt_sequence[sequence_start:sequence_end]
        if (
            not sequence
            or len(sequence) > ESM_MAX_RESIDUES
            or any(position < 0 or position >= len(sequence) for position in relative_positions)
        ):
            logger.warning(
                "[%s] cannot use ESM fallback for positions %s in sequence window [%d:%d]",
                assay.assay_id,
                positions,
                sequence_start,
                sequence_end,
            )
            continue
        fallback_groups.append(
            (
                sequence,
                relative_positions,
                mutant_indices,
            )
        )

    fallback_positions = 0
    model.eval()
    for start in range(0, len(fallback_groups), batch_size):
        batch_groups = fallback_groups[start : start + batch_size]
        sequences = [group[0] for group in batch_groups]
        plans = []
        for _, relative_positions, _ in batch_groups:
            masked = _torch.tensor(relative_positions, dtype=_torch.long, device=device)
            plans.append(
                {
                    "masked": masked,
                    "types": _torch.zeros_like(masked),
                    "targets": _torch.zeros_like(masked),
                    "random_aa": _torch.full_like(masked, -1),
                }
            )
        with _torch.no_grad():
            _, sequence_logits = model._run_esm_masked(
                sequences, plans, device, _torch.float32
            )
            sequence_logits = _torch.log_softmax(sequence_logits, dim=-1)

        cursor = 0
        for (_, relative_positions, mutant_indices), sequence in zip(
            batch_groups, sequences
        ):
            num_residues = len(sequence)
            log_probs = sequence_logits[cursor : cursor + num_residues]
            cursor += num_residues
            position_indices = _torch.tensor(relative_positions, device=device)
            for mutant_index in mutant_indices:
                mutant = assay.mutants[mutant_index]
                wt_idx = _torch.tensor(
                    [aa_one_letter_to_idx(aa) for aa in mutant.wt_aas],
                    device=device,
                )
                mt_idx = _torch.tensor(
                    [aa_one_letter_to_idx(aa) for aa in mutant.mt_aas],
                    device=device,
                )
                scores[mutant_index] = (
                    (
                        log_probs[position_indices, mt_idx]
                        - log_probs[position_indices, wt_idx]
                    )
                    .sum()
                    .item()
                )
            fallback_positions += len(relative_positions)
    return fallback_positions


def score_assay_esm2(
    module,
    assay: DMSAssay,
    device: str,
    batch_size: int = 8,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Score an assay with masked ESM-2 log-odds only, ignoring structure.

    This is the pure-sequence baseline: it reuses the exact ESM path that
    ``score_assay_alphasurf`` falls back to, so a harness bug shows up as a
    deviation from ESM-2's published 0.414 rather than being confounded with the
    structural head. Positions are already sequence coordinates here, so no
    structure offset is applied and no AF2 structure is read.
    """
    model = module.model
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for mutant_index, mutant in enumerate(assay.mutants):
        groups.setdefault(tuple(mutant.positions), []).append(mutant_index)

    scores = np.full(len(assay.mutants), np.nan, dtype=np.float64)
    scored_positions = _score_esm_fallback_groups(
        model,
        assay,
        list(groups.items()),
        scores,
        device,
        batch_size,
        assay.seq_len,
        0,
    )
    diagnostics = {
        "num_groups": len(groups),
        "num_groups_geometry_failed": 0,
        "num_positions_low_plddt": 0,
        "num_positions_esm_scored": scored_positions,
    }
    return scores, diagnostics


def score_assay_alphasurf(
    module,
    loader,
    pdb_path,
    protein_name,
    assay: DMSAssay,
    device: str,
    batch_size: int = 8,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    progress: bool = False,
    structure_length: int | None = None,
    sequence_length: int | None = None,
    structure_offset: int = 0,
    reference_protein=None,
    plddt_threshold: float | None = 70.0,
    s3f_reference=None,
) -> np.ndarray:
    """Score mutants with S3F-style log-odds from the residue head.

    For each mutant, mask the mutation positions, run the encoder + head,
    take log_softmax at masked positions, and compute:
        score = sum [log P(MT_AA) - log P(WT_AA)]

    Mutants sharing the same set of positions reuse one masked forward pass,
    as in S3F's released evaluator. For AlphaSurf, the masked residues are
    replaced with the checkpoint's configured alanine or N/CA/C/O geometry,
    and both graph and surface are regenerated to match pretraining. Long
    sequences use S3F's 1,022-residue window. At mutation sites with an AF2
    B-factor below ``plddt_threshold``, the ESM-2 logits replace the structural
    logits, matching S3F's low-confidence fallback. If masked geometry
    generation fails, the same masked ESM-2 log-odds are used for that group.
    For s3f_exact checkpoints, pass ``s3f_reference`` (an S3FReference): its
    backbone surface is reused for every group and only the sequence is masked.

    Returns a float array of length len(assay.mutants) plus a diagnostics dict
    recording how much of the assay was scored by the fallbacks rather than by
    the structural head.
    """
    import torch as _torch

    from alphasurf.utils.data_utils import AtomBatch as _AtomBatch

    model = module.model

    reference_b_factors = None
    if plddt_threshold is not None and (
        reference_protein is not None or s3f_reference is not None
    ):
        if s3f_reference is not None:
            reference_b_factors = s3f_reference.b_factor
        else:
            reference_graph = getattr(reference_protein, "graph", None)
            reference_b_factors = getattr(reference_graph, "b_factor", None)
        if reference_b_factors is not None:
            reference_b_factors = reference_b_factors.to(device)
        else:
            logger.warning(
                "[%s] AF2 B-factors unavailable; disabling pLDDT fallback",
                assay.assay_id,
            )

    if structure_length is None and s3f_reference is not None:
        structure_length = len(s3f_reference.sequence)
    if structure_length is None:
        reference = reference_protein or loader.load(protein_name, pdb_path=pdb_path)
        if reference is None or reference.graph is None:
            return np.full(len(assay.mutants), np.nan, dtype=np.float64), {}
        structure_length = int(reference.graph.x.shape[0])
    if sequence_length is None:
        sequence_length = assay.seq_len

    groups: Dict[Tuple[int, ...], List[int]] = {}
    for mutant_index, mutant in enumerate(assay.mutants):
        groups.setdefault(tuple(mutant.positions), []).append(mutant_index)

    cfg = getattr(getattr(module, "hparams", None), "cfg", None)
    structure_mask_cfg = getattr(cfg, "structure_mask", None)
    # Checkpoints predating configurable structural masking were all trained
    # with the N/CA/C/O-only behavior.
    structure_mask_mode = str(getattr(structure_mask_cfg, "mode", "backbone"))
    head_to_res_type = None
    if s3f_reference is not None:
        structure_mask_mode = "sequence"
        head_to_res_type = s3f_head_to_res_type_index().to(device)
        geometry_dataset = S3FGeometryDataset(
            reference=s3f_reference,
            assay_id=assay.assay_id,
            structure_length=structure_length,
            group_items=list(groups.items()),
            sequence_length=sequence_length,
            structure_offset=structure_offset,
        )
    else:
        geometry_dataset = MaskedGeometryDataset(
            protein_loader=loader,
            pdb_path=str(pdb_path),
            protein_name=protein_name,
            assay_id=assay.assay_id,
            structure_length=structure_length,
            group_items=list(groups.items()),
            structure_mask_mode=structure_mask_mode,
            sequence_length=sequence_length,
            structure_offset=structure_offset,
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
        "[%s] generating %d unique %s-masked geometries with %d workers",
        assay.assay_id,
        len(geometry_dataset),
        structure_mask_mode,
        num_workers,
    )

    scores = np.full(len(assay.mutants), np.nan, dtype=np.float64)
    fallback_positions = 0
    with tqdm(
        total=len(geometry_dataset),
        desc=assay.assay_id[:36],
        unit="geometry",
        position=1,
        leave=False,
        mininterval=2.0,
        dynamic_ncols=False,
        file=sys.stdout,
        disable=not progress,
    ) as geometry_progress:
        for geometry_batch, attempted_count in geometry_loader:
            geometry_progress.update(attempted_count)
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
                if head_to_res_type is not None:
                    masked_logits = masked_logits[:, head_to_res_type]
                sequence_logits = out.get("sequence_logits")
                cursor = 0
                for sample_index, (mutant_indices, num_positions) in enumerate(
                    metadata
                ):
                    structural_log_probs = _torch.log_softmax(
                        masked_logits[cursor : cursor + num_positions], dim=-1
                    )
                    cursor += num_positions
                    pos_range = _torch.arange(num_positions, device=device)
                    log_probs = structural_log_probs
                    if reference_b_factors is not None and plddt_threshold is not None:
                        structure_positions = samples[sample_index].structure_positions
                        plddt = reference_b_factors[structure_positions.to(device)]
                        low_plddt = _torch.isfinite(plddt) & (plddt < plddt_threshold)
                        if low_plddt.any():
                            if sequence_logits is None:
                                raise RuntimeError(
                                    "S3F model output is missing ESM sequence logits"
                                )
                            graph_positions = (
                                batch.graph.ptr[sample_index]
                                + samples[sample_index].masked_positions.to(device)
                            )
                            esm_log_probs = _torch.log_softmax(
                                sequence_logits[graph_positions], dim=-1
                            )
                            log_probs = _torch.where(
                                low_plddt.unsqueeze(-1),
                                esm_log_probs,
                                structural_log_probs,
                            )
                            fallback_positions += int(low_plddt.sum().item())
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
                            (
                                log_probs[pos_range, mt_idx]
                                - log_probs[pos_range, wt_idx]
                            )
                            .sum()
                            .item()
                        )

    if fallback_positions:
        logger.info(
            "[%s] used ESM logits for %d low-pLDDT masked positions",
            assay.assay_id,
            fallback_positions,
        )
    failed_groups = [
        (positions, mutant_indices)
        for positions, mutant_indices in groups.items()
        if any(np.isnan(scores[index]) for index in mutant_indices)
    ]
    if failed_groups:
        geometry_fallback_positions = _score_esm_fallback_groups(
            model,
            assay,
            failed_groups,
            scores,
            device,
            batch_size,
            sequence_length,
            structure_offset,
        )
        if geometry_fallback_positions:
            logger.info(
                "[%s] used ESM-only fallback for %d masked positions in %d failed geometries",
                assay.assay_id,
                geometry_fallback_positions,
                len(failed_groups),
            )
    diagnostics = {
        "num_groups": len(groups),
        "num_groups_geometry_failed": len(failed_groups),
        "num_positions_low_plddt": fallback_positions,
    }
    return scores, diagnostics
