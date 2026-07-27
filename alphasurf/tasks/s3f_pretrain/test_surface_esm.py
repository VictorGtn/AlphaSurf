from types import SimpleNamespace
from unittest import TestCase

import torch
from torch_geometric.data import Data

from alphasurf.tasks.s3f_pretrain.model import (
    ESM_EMBED_DIM,
    S3FPretrainNet,
    SurfaceESMInjector,
)


class SurfaceESMInjectorTest(TestCase):
    def test_projection_parameter_count_is_lightweight(self):
        for hidden_dim in (128, 256):
            injector = SurfaceESMInjector(output_dim=hidden_dim, k=3)
            parameter_count = sum(p.numel() for p in injector.parameters())
            self.assertEqual(parameter_count, (ESM_EMBED_DIM + 1) * hidden_dim)

    def test_knn_lift_never_mixes_proteins_in_overlapping_frames(self):
        # Both proteins deliberately occupy the same coordinates. If batching
        # were ignored, the surface features would mix values from both.
        residue_positions = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        )
        graph = Data(
            node_pos=residue_positions,
            batch=torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        surface = Data(
            x=torch.tensor([[7.0], [8.0]]),
            verts=torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            batch=torch.tensor([0, 1]),
        )
        esm = torch.zeros((6, ESM_EMBED_DIM))
        esm[:, 0] = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

        injector = SurfaceESMInjector(output_dim=1, k=3)
        with torch.no_grad():
            injector.esm_projection.weight.zero_()
            injector.esm_projection.weight[0, 0] = 1.0
            injector.distance_projection.weight.zero_()

        output = injector(surface, graph, esm)

        self.assertEqual(tuple(output.x.shape), (2, 2))
        self.assertTrue(torch.allclose(output.x[:, 0], torch.tensor([7.0, 8.0])))
        self.assertTrue(torch.allclose(output.x[:, 1], torch.tensor([2.0, 20.0])))

    def test_exact_encoder_ignores_alpha_surface_injection(self):
        encoder_cfg = SimpleNamespace(name="s3f_exact", blocks=[])
        head_cfg = SimpleNamespace(encoded_dims=128, dropout=0.5)
        surface_esm_cfg = SimpleNamespace(enabled=True, k=3)

        model = S3FPretrainNet(
            encoder_cfg,
            head_cfg,
            cfg_surface_esm=surface_esm_cfg,
        )

        self.assertIsNone(model.surface_esm_injector)

    def test_alpha_encoder_enables_configured_injection(self):
        encoder_cfg = SimpleNamespace(name="pronet_gvpencoder", blocks=[])
        head_cfg = SimpleNamespace(encoded_dims=128, dropout=0.5)
        surface_esm_cfg = SimpleNamespace(enabled=True, k=3)

        model = S3FPretrainNet(
            encoder_cfg,
            head_cfg,
            cfg_surface_esm=surface_esm_cfg,
        )

        self.assertIsInstance(model.surface_esm_injector, SurfaceESMInjector)
        self.assertEqual(model.surface_esm_injector.k, 3)
