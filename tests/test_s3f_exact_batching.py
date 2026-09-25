from unittest import TestCase

import torch

from alphasurf.network_utils.misc_arch.s3f_blocks import _surface_residue_knn
from alphasurf.tasks.s3f_pretrain.model import S3FPretrainNet


class S3FExactBatchingTest(TestCase):
    def test_exact_prediction_head_accepts_256_dimensional_encoder_output(self):
        class Config:
            name = "s3f_exact"
            blocks = []

        class HeadConfig:
            encoded_dims = 128
            dropout = 0.5

        model = S3FPretrainNet(Config(), HeadConfig())
        self.assertEqual(model.encoded_dim, 256)
        self.assertEqual(model.residue_head[-1].in_features, 256)

    def test_surface_residue_knn_never_crosses_proteins(self):
        # The two proteins deliberately occupy identical coordinate frames.
        # A global cdist/top-k can mix them; batch-aware PyG kNN cannot.
        res_pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        surf_pos = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [1.9, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.9, 0.0, 0.0],
            ]
        )
        res_batch = torch.tensor([0, 0, 1, 1])
        surf_batch = torch.tensor([0, 0, 1, 1])

        nn_idx, dists = _surface_residue_knn(
            res_pos,
            surf_pos,
            k=2,
            res_batch=res_batch,
            surf_batch=surf_batch,
        )

        self.assertTrue(
            torch.equal(res_batch[nn_idx], surf_batch[:, None].expand(-1, 2))
        )
        self.assertTrue(torch.allclose(dists[:, 0], torch.full((4,), 0.1)))
