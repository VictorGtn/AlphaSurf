"""
ProteinGym substitutions dataset.

Parses a single DMS assay CSV into wild-type sequence + list of mutants
with their per-position WT/MT amino acids and measured fitness scores.

The CSV schema is `mutant, mutated_sequence, DMS_score`. The `mutant` field
is either `A123V` (single-site) or colon-separated for multi-site (`A123V:D45E`).
Position numbering is 1-indexed relative to the assay's full reference
sequence. S3F applies hardcoded model-input crops for three assays while
retaining those absolute mutation positions; we preserve that behavior here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


# Zero-indexed, half-open model-input crop windows from S3F's evaluate.py.
# Mutation strings and mutated_sequence remain in full-sequence coordinates.
ASSAY_RESIDUE_RANGES: Dict[str, Tuple[int, int]] = {
    "POLG_HCVJF_Qi_2014": (1981, 2225),
    "A0A140D2T1_ZIKV_Sourisseau_2019": (290, 794),
    "B2L11_HUMAN_Dutta_2010_binding-Mcl-1": (119, 197),
}

_MUTANT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


@dataclass
class Mutant:
    """One row of a DMS assay."""

    mutant_str: str
    mutated_sequence: str
    score: float
    positions: List[int]  # 0-indexed into the full WT sequence
    wt_aas: List[str]  # WT amino acid at each position
    mt_aas: List[str]  # MT amino acid at each position


@dataclass
class DMSAssay:
    """One parsed DMS assay."""

    assay_id: str
    uniprot_id: str
    wt_sequence: str
    mutants: List[Mutant] = field(default_factory=list)

    @property
    def seq_len(self) -> int:
        return len(self.wt_sequence)

    @property
    def mutated_positions(self) -> List[int]:
        seen = set()
        positions: List[int] = []
        for m in self.mutants:
            for p in m.positions:
                if p not in seen:
                    seen.add(p)
                    positions.append(p)
        return sorted(positions)


def parse_mutant_field(mutant_str: str) -> Tuple[List[str], List[int], List[str]]:
    """Parse `A123V` or `A123V:D45E` into (wt_aas, positions_1indexed, mt_aas)."""
    wt_aas: List[str] = []
    positions: List[int] = []
    mt_aas: List[str] = []
    for token in mutant_str.split(":"):
        match = _MUTANT_RE.match(token)
        if match is None:
            raise ValueError(f"Unparsable mutant token: {token!r}")
        wt, pos, mt = match.group(1), int(match.group(2)), match.group(3)
        wt_aas.append(wt)
        positions.append(pos)
        mt_aas.append(mt)
    return wt_aas, positions, mt_aas


def derive_wt_sequence(mutants: List[Mutant], fallback_seq: str) -> str:
    """Recover the WT sequence by reverting every mutated position on the
    mutated_sequence of the first mutant. mutated_sequence length is assumed
    equal across all rows; this is checked by the caller."""
    if not mutants:
        return fallback_seq
    m = mutants[0]
    seq = list(m.mutated_sequence)
    for pos, wt in zip(m.positions, m.wt_aas):
        seq[pos] = wt
    return "".join(seq)


def load_dms_assay(csv_path: str | Path) -> DMSAssay:
    """Parse a ProteinGym DMS substitution CSV into a DMSAssay.

    The CSV is expected to have columns `mutant`, `mutated_sequence`, and
    `DMS_score`. The filename stem is used as the assay id; the `UniProt_ID`
    column (if present) overrides the uniprot id.
    """
    csv_path = Path(csv_path)
    assay_id = csv_path.stem

    df = pd.read_csv(csv_path)
    required = {"mutant", "mutated_sequence", "DMS_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    seq_len = len(df.iloc[0]["mutated_sequence"])
    seq_lens = df["mutated_sequence"].str.len()
    if (seq_lens != seq_len).any():
        raise ValueError(
            f"{csv_path}: mutated_sequence length is not constant "
            f"({seq_lens.min()}..{seq_lens.max()})"
        )

    uniprot_id = (
        df["UniProt_ID"].iloc[0]
        if "UniProt_ID" in df.columns and pd.notna(df["UniProt_ID"].iloc[0])
        else assay_id
    )

    mutants: List[Mutant] = []
    for _, row in df.iterrows():
        wt_aas, positions_1, mt_aas = parse_mutant_field(row["mutant"])
        positions_0 = [p - 1 for p in positions_1]
        for wt, pos_0, mt, p1 in zip(wt_aas, positions_0, mt_aas, positions_1):
            if pos_0 < 0 or pos_0 >= seq_len:
                raise ValueError(
                    f"{csv_path}: mutant {row['mutant']!r} position {p1} "
                    f"out of range for sequence length {seq_len}"
                )
            if row["mutated_sequence"][pos_0] != mt:
                raise ValueError(
                    f"{csv_path}: mutant {row['mutant']!r} expects {mt} at "
                    f"position {p1} but mutated_sequence has "
                    f"{row['mutated_sequence'][pos_0]}"
                )
        mutants.append(
            Mutant(
                mutant_str=row["mutant"],
                mutated_sequence=row["mutated_sequence"],
                score=float(row["DMS_score"]),
                positions=positions_0,
                wt_aas=wt_aas,
                mt_aas=mt_aas,
            )
        )

    fallback = df.iloc[0]["mutated_sequence"]
    wt_sequence = derive_wt_sequence(mutants, fallback)

    for wt_aa, pos_0 in zip(mutants[0].wt_aas, mutants[0].positions):
        if wt_sequence[pos_0] != wt_aa:
            raise ValueError(
                f"{csv_path}: derived WT has {wt_sequence[pos_0]} at position "
                f"{pos_0 + 1}, mutant expected {wt_aa}"
            )

    return DMSAssay(
        assay_id=assay_id,
        uniprot_id=str(uniprot_id),
        wt_sequence=wt_sequence,
        mutants=mutants,
    )


def list_assay_csvs(substitutions_dir: str | Path) -> List[Path]:
    """Return every DMS CSV under `substitutions_dir`, sorted."""
    substitutions_dir = Path(substitutions_dir)
    csvs = sorted(substitutions_dir.rglob("*.csv"))
    return [p for p in csvs if p.stem != "ProteinGym_info"]


def af2_structure_path(af2_dir: str | Path, uniprot_id: str) -> Optional[Path]:
    """Resolve the AF2 PDB for a UniProt id. Returns None if not found."""
    af2_dir = Path(af2_dir)
    candidates = [
        af2_dir / f"{uniprot_id}.pdb",
        af2_dir / "AF2_structures" / f"{uniprot_id}.pdb",
        af2_dir / f"{uniprot_id}.pdb.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fall back to a unique-prefix match if the layout is different.
    matches = list(af2_dir.rglob(f"{uniprot_id}*.pdb"))
    if len(matches) == 1:
        return matches[0]
    return None


class ProteinGymDataset(Dataset):
    """Thin Dataset wrapper over a parsed DMSAssay.

    Each item is one mutant row. The WT structure / sequence is loaded once by
    the caller (in evaluate.py) and shared across mutants in the same assay;
    this class only handles the per-row bookkeeping.
    """

    def __init__(self, csv_path: str | Path):
        self.assay = load_dms_assay(csv_path)

    def __len__(self) -> int:
        return len(self.assay.mutants)

    def __getitem__(self, idx: int) -> Mutant:
        return self.assay.mutants[idx]


def targets_and_predictions(
    assay: DMSAssay, predicted_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack DMS_score column against model predictions for metric computation."""
    if predicted_scores.shape[0] != len(assay.mutants):
        raise ValueError(
            f"predictions length {predicted_scores.shape[0]} does not match "
            f"assay size {len(assay.mutants)}"
        )
    targets = np.array([m.score for m in assay.mutants], dtype=np.float64)
    return targets, np.asarray(predicted_scores, dtype=np.float64)
