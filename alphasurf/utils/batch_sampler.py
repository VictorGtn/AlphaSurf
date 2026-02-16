"""
Dynamic batch sampler for variable-size meshes.

Uses atom count as a proxy for mesh size to create batches with
a target total "budget", ensuring efficient GPU utilization.
"""

import random
import bisect
from typing import Iterator, List, Optional

import pandas as pd
from torch.utils.data import Sampler


class AtomBudgetBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that groups samples by total atom count budget using First-Fit-Decreasing (FFD).

    Uses atom count as a proxy for mesh vertex count.
    Implements FFD to strictly respect max_atoms budget to prevent OOM,
    which may result in batches smaller than min_batch_size for very large proteins.

    Args:
        sizes: List of atom counts for each sample
        max_atoms: Maximum total atoms per batch
        min_batch_size: Target minimum samples per batch (soft constraint in FFD)
        shuffle: Whether to shuffle the order of batches each epoch
        drop_last: Ignored in FFD implementation (all samples needed for optimal packing)
    """

    def __init__(
        self,
        sizes: List[int],
        max_atoms: int = 50000,
        min_batch_size: int = 2,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.sizes = sizes
        self.max_atoms = max_atoms
        self.min_batch_size = min_batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Pre-compute batches using FFD
        # FFD is deterministic for a given set of sizes, so we compute once.
        indices = list(range(len(self.sizes)))
        all_batches = self._create_batches(indices)

        # Filter batches with size < min_batch_size
        # The user specifically asked to drop batches of size 1 (or < min_batch_size)
        self._batches = []
        dropped_batches = 0
        dropped_samples = 0

        for batch in all_batches:
            if len(batch) < self.min_batch_size:
                dropped_batches += 1
                dropped_samples += len(batch)
            else:
                self._batches.append(batch)

        # Print statistics
        total_samples = len(self.sizes)
        if total_samples > 0:
            dropped_pct = (dropped_samples / total_samples) * 100.0
        else:
            dropped_pct = 0.0

        print("AtomBudgetBatchSampler (FFD):")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Dropped {dropped_batches} batches (size < {self.min_batch_size})")
        print(f"  - Dropped {dropped_samples} samples ({dropped_pct:.2f}% of dataset)")
        print(f"  - Retained {len(self._batches)} valid batches")

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        size_columns: List[str] = ["n_atoms_R", "n_atoms_L"],
        **kwargs,
    ) -> "AtomBudgetBatchSampler":
        """
        Create sampler from a CSV file with atom count columns.

        Args:
            csv_path: Path to CSV file
            size_columns: Column names to sum for total size per sample
            **kwargs: Additional arguments for AtomBudgetBatchSampler
        """
        df = pd.read_csv(csv_path)
        sizes = df[size_columns].sum(axis=1).tolist()
        return cls(sizes=sizes, **kwargs)

    @classmethod
    def from_pdb_dir(
        cls, systems: List[dict], pdb_dir: str, **kwargs
    ) -> "AtomBudgetBatchSampler":
        """
        Create sampler by counting atoms in PDB files.

        Args:
            systems: List of system dicts with 'receptor_id' and 'ligand_id'
            pdb_dir: Directory containing PDB files
            **kwargs: Additional arguments for AtomBudgetBatchSampler
        """
        sizes = cls.get_system_sizes(systems, pdb_dir)
        return cls(sizes=sizes, **kwargs)

    @staticmethod
    def get_system_sizes(
        systems: List[dict], pdb_dir: str, cache_path: Optional[str] = None
    ) -> List[int]:
        """
        Compute atom counts for a list of systems.
        If cache_path is provided and does not exist, dump the results to json.
        Does NOT load from cache (always computes).
        """
        import os
        import json

        def count_atoms(pdb_path: str) -> int:
            if not os.path.exists(pdb_path):
                return 0
            count = 0
            with open(pdb_path, "r") as f:
                for line in f:
                    if line.startswith("ATOM"):
                        count += 1
            return count

        sizes = []
        for sys in systems:
            r_path = os.path.join(pdb_dir, f"{sys['receptor_id']}.pdb")
            l_path = os.path.join(pdb_dir, f"{sys['ligand_id']}.pdb")
            total = count_atoms(r_path) + count_atoms(l_path)
            sizes.append(total)

        if cache_path is not None:
            if not os.path.exists(cache_path):
                print(f"Dumping atom counts to {cache_path}")
                try:
                    os.makedirs(
                        os.path.dirname(os.path.abspath(cache_path)), exist_ok=True
                    )
                    with open(cache_path, "w") as f:
                        json.dump(sizes, f)
                except Exception as e:
                    print(f"Failed to dump atom counts: {e}")
            else:
                print(
                    f"Atom counts cache already exists at {cache_path}, skipping dump."
                )

        return sizes

    def _create_batches(self, indices: List[int]) -> List[List[int]]:
        """
        Create batches using First-Fit-Decreasing (FFD) algorithm.
        Strictly respects max_atoms budget to avoid OOM.
        """
        # Prepare items as (size, index) tuples
        items = [(self.sizes[i], i) for i in indices]

        # Sort by size ascending (so we can pop largest from end)
        items.sort(key=lambda x: x[0])

        # Separate into lists for bisect efficiency
        sorted_sizes = [x[0] for x in items]
        sorted_indices = [x[1] for x in items]

        batches = []

        while sorted_sizes:
            # 1. Start new batch with the largest available item
            current_size = sorted_sizes.pop()
            current_idx = sorted_indices.pop()
            batch = [current_idx]

            space_left = self.max_atoms - current_size

            # 2. Fill remaining space with largest items that fit
            while space_left > 0 and sorted_sizes:
                # Find insertion point for space_left
                # bisect_right returns index where all elements to left are <= space_left
                idx = bisect.bisect_right(sorted_sizes, space_left)

                if idx == 0:
                    # No item fits
                    break

                # The largest item that fits is at idx - 1
                target_idx = idx - 1

                # Add to batch and remove from pool
                s = sorted_sizes.pop(target_idx)
                i = sorted_indices.pop(target_idx)

                batch.append(i)
                space_left -= s

            batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        # FFD batches are deterministic in composition.
        # We introduce randomness by shuffling the order of batches.
        batches = list(self._batches)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return len(self._batches)
