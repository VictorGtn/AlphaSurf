from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
import torch

from alphasurf.tasks.s3f_pretrain import sampling


class SamplingTest(TestCase):
    def test_split_matches_s3f_random_split_permutation(self):
        items = [f"protein_{i}" for i in range(100)]
        expected_indices = torch.randperm(
            len(items), generator=torch.Generator().manual_seed(0)
        ).tolist()
        expected = [items[i] for i in expected_indices]

        train = sampling.split_like_s3f(items, "train")
        val = sampling.split_like_s3f(items, "val")
        test = sampling.split_like_s3f(items, "test")

        self.assertEqual(train + val + test, expected)
        self.assertEqual((len(train), len(val), len(test)), (97, 2, 1))

    def test_mask_count_uses_floor(self):
        self.assertEqual(sampling.masked_residue_count(250, 0.15), 37)
        self.assertEqual(sampling.masked_residue_count(3, 0.15), 1)

    def test_worker_rng_uses_distinct_pytorch_worker_seeds(self):
        dataset = SimpleNamespace(seed=2024, _rng=None, _rng_seed=None)

        with mock.patch.object(
            sampling,
            "get_worker_info",
            return_value=SimpleNamespace(seed=101, id=0),
        ):
            worker_zero = sampling.worker_rng(dataset).integers(0, 2**31, size=8)

        with mock.patch.object(
            sampling,
            "get_worker_info",
            return_value=SimpleNamespace(seed=102, id=1),
        ):
            worker_one = sampling.worker_rng(dataset).integers(0, 2**31, size=8)

        expected_zero = np.random.default_rng(101).integers(0, 2**31, size=8)
        self.assertTrue(np.array_equal(worker_zero, expected_zero))
        self.assertFalse(np.array_equal(worker_zero, worker_one))
