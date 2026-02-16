import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict

from .dmasif_utils.benchmark_models import dMaSIFConv_seg
from .dmasif_utils.geometry_processing import curvatures


class dMasifWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, argdir):
        super().__init__()
        with open(
            argdir, "r"
        ) as f:  # "alphasurf/alphasurf/tasks/shared_conf/config.yml"
            dmasifcfg = EasyDict(yaml.safe_load(f))
        self.args = dmasifcfg.model.dmasif
        self.curvature_scales = self.args.curvature_scales
        self.orientation_scores = nn.Sequential(
            nn.Linear(dim_in + 2 * (len(self.curvature_scales)), dim_out),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(dim_out, 1),
        )
        self.conv = dMaSIFConv_seg(
            self.args,
            in_channels=dim_in + 2 * (len(self.curvature_scales)),
            out_channels=dim_out,
            n_layers=self.args.n_layers,
            radius=self.args.radius,
        )
        self.dim_out = dim_out

    def features(self, surface):
        """Estimates geometric and chemical features from a protein surface or a cloud of atoms."""
        cached = getattr(surface, "cached_curvatures", None)
        if cached is None:
            P_curvatures = curvatures(
                surface.verts,
                triangles=None,
                normals=surface.vnormals,
                scales=self.curvature_scales,
                batch=surface.batch,
            )
            surface.cached_curvatures = P_curvatures
        else:
            P_curvatures = cached
        surface.x = torch.cat([surface.x, P_curvatures], dim=1)
        return surface

    def _dummy_transform(self, surface, out_dim):
        in_dim = surface.x.shape[-1]
        if in_dim == out_dim:
            return surface
        elif in_dim < out_dim:
            surface.x = F.pad(surface.x, (0, out_dim - in_dim))
        else:
            surface.x = surface.x[..., :out_dim]
        return surface

    def forward(self, surface):
        dummy_enc = False
        if dummy_enc:
            return self._dummy_transform(surface, self.dim_out)
        # Validate input
        if surface.verts is None or surface.verts.numel() == 0:
            raise ValueError("Surface vertices are empty or None")
        if surface.vnormals is None or surface.vnormals.numel() == 0:
            raise ValueError("Surface normals are empty or None")
        if surface.x is None or surface.x.numel() == 0:
            raise ValueError("Surface features are empty or None")

        surface = self.features(surface)

        weights = self.orientation_scores(surface.x)
        self.conv.load_mesh(
            xyz=surface.verts,
            normals=surface.vnormals,
            weights=weights,
            batch=surface.batch,
        )
        surface.x = self.conv(surface.x)
        return surface
