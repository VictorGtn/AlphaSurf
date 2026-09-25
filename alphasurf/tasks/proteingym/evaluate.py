#!/usr/bin/env python3
"""
Zero-shot ProteinGym evaluation.

alphasurf: S3F-style log-odds from the masked residue head (S3F checkpoint).
esm2:      masked ESM-2 log-odds only, no structure (sequence baseline).

Usage:
    python -m alphasurf.tasks.proteingym.evaluate \
        --ckpt alphasurf/tasks/s3f_pretrain/ckpt/last.ckpt \
        --scoring-method alphasurf \
        --substitutions-dir data/proteingym/substitutions/DMS_ProteinGym_substitutions \
        --af2-dir data/proteingym/af2_structures/ProteinGym_AF2_structures \
        --output-dir runs/proteingym_s3f
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.tasks.proteingym.dataset import (
    ASSAY_RESIDUE_RANGES,
    DMSAssay,
    af2_structure_path,
    find_reference_file,
    list_assay_csvs,
    load_dms_assay,
    load_reference_metadata,
)
from alphasurf.tasks.proteingym.model import (
    build_protein_loader,
    load_s3f_module,
    load_s3f_reference,
)
from alphasurf.tasks.proteingym.scoring import (
    RES_TYPE_ONEHOT_SLICE,
    aa_one_letter_to_idx,
    compute_metrics,
    score_assay_esm2,
    score_assay_alphasurf,
)

logger = logging.getLogger("alphasurf.proteingym")


def graph_aa_sequence(graph) -> np.ndarray:
    """Per-node AA res_type_dict index from the graph's residue-type one-hot."""
    onehot = graph.x[:, RES_TYPE_ONEHOT_SLICE]
    return onehot.argmax(dim=-1).cpu().numpy()


def resolve_position_offset(assay: DMSAssay, af2_graph_len: int) -> Optional[int]:
    """Offset to add to full-sequence mutant positions to index the AF2 graph.

    Returns None if the assay cannot be mapped to the structure unambiguously.
    """
    if af2_graph_len == assay.seq_len:
        return 0

    residue_range = assay.structure_range or ASSAY_RESIDUE_RANGES.get(assay.assay_id)
    if residue_range is None:
        logger.warning(
            f"[{assay.assay_id}] AF2 graph length {af2_graph_len} != WT "
            f"sequence length {assay.seq_len} and no residue range is "
            f"registered; skipping."
        )
        return None
    start, end = residue_range
    if af2_graph_len == end - start:
        return -start
    if af2_graph_len >= end:
        return 0
    logger.warning(
        f"[{assay.assay_id}] AF2 graph length {af2_graph_len} matches neither "
        f"the full sequence ({assay.seq_len}) nor S3F crop [{start}:{end}] "
        f"({end - start} residues); skipping."
    )
    return None


def validate_aa_identity(assay: DMSAssay, aa_idx: np.ndarray, offset: int) -> bool:
    """Sanity-check that AF2 residue types (res_type_dict indices) match the WT
    amino acids in the CSV at the first mutant's positions. Returns True if all
    match."""
    m = assay.mutants[0]
    for pos, wt_aa in zip(m.positions, m.wt_aas):
        graph_pos = pos + offset
        if graph_pos < 0 or graph_pos >= len(aa_idx):
            logger.warning(
                f"[{assay.assay_id}] mutant position {pos}+{offset} out of "
                f"range for AF2 graph of length {len(aa_idx)}."
            )
            return False
        expected_idx = aa_one_letter_to_idx(wt_aa)
        if aa_idx[graph_pos] != expected_idx:
            logger.warning(
                f"[{assay.assay_id}] AF2 residue at {graph_pos} is "
                f"{aa_idx[graph_pos]} but CSV WT is {wt_aa} "
                f"({expected_idx})."
            )
            return False
    return True


def score_one_assay(
    csv_path: Path,
    af2_dir: Path,
    module,
    loader,
    device: str,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    progress: bool,
    scoring_method: str,
    metadata: Optional[dict] = None,
    plddt_threshold: float | None = 70.0,
    s3f_surface_dir: Optional[Path] = None,
) -> Optional[dict]:
    assay = load_dms_assay(csv_path)
    structure_id = assay.uniprot_id
    if metadata is not None:
        assay.uniprot_id = str(metadata["UniProt_ID"])
        structure_id = str(metadata["pdb_file"])
        pdb_range = str(metadata["pdb_range"])
        start, end = (int(value) for value in pdb_range.split("-"))
        assay.structure_range = (start - 1, end)

    if scoring_method == "esm2":
        predictions, diagnostics = score_assay_esm2(
            module, assay, device, batch_size=batch_size
        )
        metrics = compute_metrics(
            np.array([m.score for m in assay.mutants], dtype=np.float64), predictions
        )
        return {
            "assay": assay,
            "predictions": predictions,
            "metrics": {**metrics, **diagnostics},
        }

    pdb_path = af2_structure_path(af2_dir, structure_id)
    if pdb_path is None:
        logger.warning(f"[{assay.assay_id}] no AF2 structure {structure_id}; skipping.")
        return None

    ref_protein = None
    s3f_reference = None
    if s3f_surface_dir is not None:
        s3f_reference = load_s3f_reference(pdb_path, s3f_surface_dir)
        built = s3f_reference is not None
    else:
        ref_protein = loader.load(assay.assay_id, pdb_path=str(pdb_path))
        built = ref_protein is not None
    if not built:
        logger.warning(
            f"[{assay.assay_id}] failed to build protein from {pdb_path}"
        )
        return None
    if s3f_reference is not None:
        aa_idx = np.array([aa_one_letter_to_idx(aa) for aa in s3f_reference.sequence])
    else:
        aa_idx = graph_aa_sequence(ref_protein.graph)
    structure_length = len(aa_idx)
    offset = resolve_position_offset(assay, structure_length)
    if offset is None:
        return None
    if not validate_aa_identity(assay, aa_idx, offset):
        logger.warning(f"[{assay.assay_id}] AA identity check failed; skipping.")
        return None
    if offset != 0:
        shifted_mutants = []
        for m in assay.mutants:
            shifted_mutants.append(
                type(m)(
                    mutant_str=m.mutant_str,
                    mutated_sequence=m.mutated_sequence,
                    score=m.score,
                    positions=[p + offset for p in m.positions],
                    wt_aas=m.wt_aas,
                    mt_aas=m.mt_aas,
                )
            )
        assay.mutants = shifted_mutants
    predictions, diagnostics = score_assay_alphasurf(
        module,
        loader,
        str(pdb_path),
        assay.assay_id,
        assay,
        device,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        progress=progress,
        structure_length=structure_length,
        sequence_length=assay.seq_len,
        structure_offset=offset,
        reference_protein=ref_protein,
        plddt_threshold=plddt_threshold,
        s3f_reference=s3f_reference,
    )
    metrics = compute_metrics(
        np.array([m.score for m in assay.mutants], dtype=np.float64), predictions
    )
    return {
        "assay": assay,
        "predictions": predictions,
        "metrics": {**metrics, **diagnostics},
    }


def write_per_assay_csv(output_dir: Path, result: dict) -> None:
    assay = result["assay"]
    preds = result["predictions"]
    df = pd.DataFrame(
        {
            "mutant": [m.mutant_str for m in assay.mutants],
            "mutated_sequence": [m.mutated_sequence for m in assay.mutants],
            "DMS_score": [m.score for m in assay.mutants],
            "model_score": preds,
        }
    )
    df.to_csv(output_dir / f"{assay.assay_id}.csv", index=False)


def write_summary_csv(output_dir: Path, summary_rows: List[dict]) -> None:
    df = pd.DataFrame(summary_rows)
    df.to_csv(output_dir / "summary.csv", index=False)


def aggregate_stats(summary_rows: List[dict]) -> dict:
    if not summary_rows:
        return {}
    spears = np.array(
        [r["spearmanr"] for r in summary_rows if not np.isnan(r["spearmanr"])]
    )
    return {
        "n_assays": len(summary_rows),
        "mean_spearman": float(spears.mean()) if len(spears) else float("nan"),
        "median_spearman": float(np.median(spears)) if len(spears) else float("nan"),
        "std_spearman": float(spears.std()) if len(spears) else float("nan"),
    }


def proteingym_aggregate(summary_rows: List[dict]) -> dict:
    """The published ProteinGym headline metric.

    Averages per UniProt first (so proteins with several assays do not dominate),
    then per coarse function category, then unweighted across the five
    categories. A flat mean over assays runs ~0.02-0.03 higher because the assay
    list is skewed towards Stability and OrganismalFitness.
    """
    by_protein: dict = {}
    for row in summary_rows:
        category = row.get("coarse_selection_type")
        if np.isnan(row["spearmanr"]) or not isinstance(category, str):
            continue
        by_protein.setdefault((category, row["UniProt_ID"]), []).append(
            row["spearmanr"]
        )
    if not by_protein:
        return {}
    by_category: dict = {}
    for (category, _), values in by_protein.items():
        by_category.setdefault(category, []).append(float(np.mean(values)))
    per_category = {
        category: float(np.mean(values))
        for category, values in sorted(by_category.items())
    }
    return {
        "proteingym_spearman": float(np.mean(list(per_category.values()))),
        "n_proteins": len(by_protein),
        "per_category": per_category,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--scoring-method",
        default="alphasurf",
        choices=["alphasurf", "esm2"],
        help="alphasurf: S3F-style masked log-odds from the residue head. esm2: masked ESM-2 log-odds only, no structure (sequence baseline, expect ~0.414).",
    )
    parser.add_argument("--substitutions-dir", required=True)
    parser.add_argument("--af2-dir", required=True)
    parser.add_argument(
        "--reference-file",
        default=None,
        help="ProteinGym DMS_substitutions.csv (auto-detected when omitted).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--assay-id",
        dest="assay_ids",
        action="append",
        default=None,
        help="Score only the named assay; repeat for multiple assays.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Worker processes for on-the-fly graph/surface generation.",
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--plddt-threshold",
        type=float,
        default=70.0,
        help="Use ESM logits for mutation sites below this AF2 pLDDT; set to -1 to disable.",
    )
    parser.add_argument(
        "--s3f-surface-dir",
        default=None,
        help="Precomputed S3F surfaces of the AF2 structures, of the type the s3f_exact checkpoint was trained on. Required for s3f_exact checkpoints.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show global assay and per-assay geometry progress bars.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    torch.manual_seed(0)
    np.random.seed(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    module, device = load_s3f_module(args.ckpt)
    s3f_exact = module.model.is_s3f_exact
    if args.scoring_method == "alphasurf" and s3f_exact != (args.s3f_surface_dir is not None):
        raise ValueError("--s3f-surface-dir is required for, and only valid with, s3f_exact checkpoints")
    loader = None if s3f_exact else build_protein_loader(module)

    reference_file = (
        Path(args.reference_file)
        if args.reference_file is not None
        else find_reference_file(args.substitutions_dir)
    )
    metadata_by_assay = {}
    if reference_file is not None:
        metadata_by_assay = load_reference_metadata(reference_file)
        logger.info(f"Loaded ProteinGym metadata from {reference_file}")
    else:
        logger.warning(
            "DMS_substitutions.csv not found; inferring structure names from assay IDs."
        )

    csv_paths = list_assay_csvs(args.substitutions_dir)
    if args.assay_ids is not None:
        csv_by_assay = {path.stem: path for path in csv_paths}
        missing = sorted(set(args.assay_ids) - csv_by_assay.keys())
        if missing:
            raise ValueError(f"Requested assays not found: {', '.join(missing)}")
        csv_paths = [csv_by_assay[assay_id] for assay_id in args.assay_ids]
    if args.limit is not None:
        csv_paths = csv_paths[: args.limit]
    logger.info(f"Scoring {len(csv_paths)} assays with {args.scoring_method}")
    plddt_threshold = None if args.plddt_threshold < 0 else args.plddt_threshold

    summary_rows: List[dict] = []
    skipped_assays: List[str] = []
    assay_iterator = tqdm(
        csv_paths,
        desc="ProteinGym assays",
        unit="assay",
        position=0,
        leave=True,
        mininterval=2.0,
        dynamic_ncols=False,
        file=sys.stdout,
        disable=not args.progress,
    )
    for i, csv_path in enumerate(assay_iterator, 1):
        logger.info(f"[{i}/{len(csv_paths)}] {csv_path.stem}")
        metadata = metadata_by_assay.get(csv_path.stem)
        result = score_one_assay(
            csv_path,
            Path(args.af2_dir),
            module,
            loader,
            device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            progress=args.progress,
            scoring_method=args.scoring_method,
            metadata=metadata,
            plddt_threshold=plddt_threshold,
            s3f_surface_dir=args.s3f_surface_dir,
        )
        if result is None:
            skipped_assays.append(csv_path.stem)
            continue
        write_per_assay_csv(output_dir, result)
        assay = result["assay"]
        metrics = result["metrics"]
        summary_rows.append(
            {
                "DMS_id": assay.assay_id,
                "UniProt_ID": assay.uniprot_id,
                "seq_len": assay.seq_len,
                "DMS_number_single_mutants": len(assay.mutants),
                "coarse_selection_type": (
                    metadata.get("coarse_selection_type") if metadata else None
                ),
                **metrics,
            }
        )

    write_summary_csv(output_dir, summary_rows)
    if skipped_assays:
        logger.warning(
            "Skipped %d/%d assays entirely (no predictions written): %s",
            len(skipped_assays),
            len(csv_paths),
            ", ".join(skipped_assays),
        )
    stats = aggregate_stats(summary_rows)
    logger.info(f"Aggregate: {stats}")
    leaderboard = proteingym_aggregate(summary_rows)
    if leaderboard:
        logger.info(
            "ProteinGym leaderboard Spearman: %.4f (%d proteins)",
            leaderboard["proteingym_spearman"],
            leaderboard["n_proteins"],
        )
        for category, value in leaderboard["per_category"].items():
            logger.info("  %-18s %.4f", category, value)
    else:
        logger.warning(
            "No coarse_selection_type metadata; skipping ProteinGym leaderboard metric"
        )


if __name__ == "__main__":
    main()
