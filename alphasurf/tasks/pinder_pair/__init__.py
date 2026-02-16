"""
PINDER-Pair task: Protein-protein interaction prediction.

Predicts interface residues between protein pairs using the PINDER dataset.
"""

from alphasurf.tasks.pinder_pair.dataset import PinderPairDataset
from alphasurf.tasks.pinder_pair.datamodule import PinderPairDataModule
from alphasurf.tasks.pinder_pair.model import PinderPairNet
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule

__all__ = [
    "PinderPairDataset",
    "PinderPairDataModule",
    "PinderPairNet",
    "PinderPairModule",
]
