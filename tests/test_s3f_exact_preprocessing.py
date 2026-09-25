import unittest

import torch

from alphasurf.tasks.s3f_pretrain.dataset_s3f_exact import CATHDatasetS3FExact
from alphasurf.tasks.s3f_pretrain.precompute_s3f_exact import (
    HKS_EIGS_RATIO,
    HKS_LARGE_EIGS_RATIO,
    HKS_LARGE_SURFACE,
)


class TestS3FExactPreprocessing(unittest.TestCase):
    def test_crop_keeps_only_referenced_surface_and_remaps(self):
        pos = torch.arange(30, dtype=torch.float32).view(10, 3)
        normals = -pos
        feat = torch.arange(420, dtype=torch.float32).view(10, 42)
        mapping = torch.tensor(
            [
                [[7, 2, 7], [5, 2, 5], [9, 5, 2]],
                [[2, 5, 7], [9, 2, 9], [5, 7, 9]],
            ]
        )

        cropped_pos, cropped_normals, cropped_feat, remapped = (
            CATHDatasetS3FExact._crop_surface(pos, normals, feat, mapping)
        )

        kept = torch.tensor([2, 5, 7, 9])
        self.assertTrue(torch.equal(cropped_pos, pos[kept]))
        self.assertTrue(torch.equal(cropped_normals, normals[kept]))
        self.assertTrue(torch.equal(cropped_feat, feat[kept]))
        self.assertEqual(tuple(remapped.shape), (2, 9))
        self.assertTrue(torch.equal(cropped_pos[remapped[0, 0]], pos[7]))
        self.assertLess(int(remapped.max()), len(kept))

    def test_released_s3f_mapping_has_63_indices_per_residue(self):
        bb_pos = torch.arange(18, dtype=torch.float32).view(2, 3, 3)
        surf_pos = torch.arange(90, dtype=torch.float32).view(30, 3)
        mapping = CATHDatasetS3FExact._compute_res2surf(bb_pos, surf_pos)
        self.assertEqual(tuple(mapping.shape), (2, 63))

    def test_large_surface_ratio_matches_released_preprocessing(self):
        self.assertEqual(HKS_LARGE_SURFACE, 20_000)
        self.assertEqual(HKS_EIGS_RATIO, 0.06)
        self.assertEqual(HKS_LARGE_EIGS_RATIO, 0.01)


if __name__ == "__main__":
    unittest.main()
