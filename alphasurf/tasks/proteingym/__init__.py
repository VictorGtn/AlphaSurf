"""
ProteinGym task: zero-shot protein fitness prediction benchmark.
"""

from alphasurf.tasks.proteingym.dataset import (
    DMSAssay,
    Mutant,
    ProteinGymDataset,
    load_dms_assay,
)

__all__ = [
    "DMSAssay",
    "Mutant",
    "ProteinGymDataset",
    "load_dms_assay",
]
