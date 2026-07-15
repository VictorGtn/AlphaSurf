"""Deterministic splitting and worker-safe randomness for S3F pretraining."""

from __future__ import annotations

from typing import Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import get_worker_info


T = TypeVar("T")
S3F_SPLIT_SEED = 0


def split_like_s3f(
    items: Sequence[T], split: str, ratios=(0.97, 0.02, 0.01)
) -> list[T]:
    """Apply S3F's seed-0 random permutation, then select one split."""
    n = len(items)
    generator = torch.Generator().manual_seed(S3F_SPLIT_SEED)
    permutation = torch.randperm(n, generator=generator).tolist()
    shuffled = [items[i] for i in permutation]

    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    if split == "train":
        return shuffled[:n_train]
    if split == "val":
        return shuffled[n_train : n_train + n_val]
    if split == "test":
        return shuffled[n_train + n_val :]
    raise ValueError(f"split must be train/val/test, got {split}")


def masked_residue_count(n_res: int, mask_rate: float) -> int:
    """S3F mask count: floor(rate * length), with at least one residue."""
    return min(max(1, int(n_res * mask_rate)), n_res)


def worker_rng(dataset) -> np.random.Generator:
    """Return a persistent RNG unique to the current DataLoader worker.

    PyTorch assigns each worker a deterministic, distinct seed.  Caching the
    generator preserves its stream with persistent workers while avoiding the
    identical NumPy Generator state produced by forking the dataset object.
    """
    worker = get_worker_info()
    worker_seed = int(dataset.seed if worker is None else worker.seed)
    if dataset._rng is None or dataset._rng_seed != worker_seed:
        dataset._rng = np.random.default_rng(worker_seed)
        dataset._rng_seed = worker_seed
    return dataset._rng
