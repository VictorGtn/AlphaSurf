import os
import sys

import torch
import torch.nn as nn

# PoissonNet has no setup.py — add repo root to sys.path
_poissonnet_dir = os.environ.get(
    "POISSONNET_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "poissonnet"),
)
_poissonnet_dir = os.path.abspath(_poissonnet_dir)
if _poissonnet_dir not in sys.path:
    sys.path.insert(0, _poissonnet_dir)

from networks.PoissonNet import PoissonBlock


class PoissonNetBlock(nn.Module):
    """
    Wrapper around PoissonNet's PoissonBlock matching DiffusionNetBlock's API:
        forward(surface) -> surface with updated .x

    Expects surface to have precomputed Poisson operators:
        surface.poisson_vert_mass, surface.poisson_L (sparse Laplacian),
        surface.poisson_G, surface.poisson_M

    The Cholesky solver is built lazily from poisson_L on first use
    (CholeskySolverF is not picklable, so we store the Laplacian instead).
    """

    def __init__(self, C_width, width=None, dropout=0.0, config=None):
        super().__init__()
        self.C_width = C_width
        width = width or C_width
        config = config or {}
        config.setdefault("dropout_p", dropout)

        self.block = PoissonBlock(
            in_c=C_width,
            out_c=C_width,
            width=width,
            config=config,
        )

    def _build_solver(self, L, device):
        from alphasurf.protein.create_operators import build_poisson_solver

        return build_poisson_solver(L, device)

    def forward(self, surface):
        is_batched = hasattr(surface, "batch") and surface.batch is not None

        if not is_batched:
            return self._forward_single(surface)
        return self._forward_batched(surface)

    def _forward_single(self, surface):
        device = surface.x.device
        solver = self._build_solver(surface.poisson_L, device)

        x_in = surface.x.unsqueeze(0)  # (V, C) -> (1, V, C)

        x_out, _ = self.block(
            x_in,
            surface.poisson_M.to(device),
            surface.poisson_G.to(device),
            [solver],
            surface.faces.unsqueeze(0).to(device=device),
            surface.poisson_vert_mass.to(device),
        )

        surface.x = x_out.squeeze(0)  # (1, V, C) -> (V, C)
        return surface

    def _forward_batched(self, surface):
        poisson_keys = ["poisson_vert_mass", "poisson_L", "poisson_G", "poisson_M"]
        poisson_lists = {k: getattr(surface, k, []) for k in poisson_keys}

        ptr = surface.ptr
        B = ptr.shape[0] - 1
        device = surface.x.device

        x_flat = surface.x
        C = x_flat.shape[1]

        vert_counts = torch.diff(ptr).tolist()

        faces_flat = surface.faces
        face_ptr = [0]
        for i in range(B):
            pm = poisson_lists.get("poisson_M", [])
            if i < len(pm):
                face_ptr.append(face_ptr[-1] + pm[i].shape[1] // 2)
            else:
                face_ptr.append(face_ptr[-1])
        face_counts = [face_ptr[i + 1] - face_ptr[i] for i in range(B)]

        # Inner-product features cannot use zero-area padded vertices.
        if self.block.inner_prod_features:
            x_outputs = []
            for i in range(B):
                V = vert_counts[i]
                F = face_counts[i]
                v_start = int(ptr[i])
                v_end = int(ptr[i + 1])
                f_start = face_ptr[i]
                f_end = face_ptr[i + 1]

                solver = self._build_solver(poisson_lists["poisson_L"][i], device)
                G = poisson_lists["poisson_G"][i].to(device)
                if G.is_sparse:
                    G = G.to_dense()
                G = G[:, : 2 * F, :V]
                x_out, _ = self.block(
                    x_flat[v_start:v_end].unsqueeze(0),
                    poisson_lists["poisson_M"][i][:, : 2 * F].to(device),
                    G,
                    [solver],
                    faces_flat[f_start:f_end].unsqueeze(0).to(device=device),
                    poisson_lists["poisson_vert_mass"][i][:, :V].to(device),
                )
                x_outputs.append(x_out.squeeze(0))

            surface.x = torch.cat(x_outputs, dim=0)
            return surface

        max_V = max(vert_counts)
        max_F = max(face_counts) if face_counts else 0

        x_padded = torch.zeros(B, max_V, C, device=device)
        faces_padded = torch.zeros(B, max_F, 3, dtype=torch.long, device=device)
        mass_padded = torch.zeros(B, max_V, device=device)
        M_padded = torch.zeros(B, 2 * max_F, device=device)
        G_padded = torch.zeros(B, 2 * max_F, max_V, device=device)
        solver_list = []

        for i in range(B):
            V = vert_counts[i]
            F = face_counts[i]
            v_start = int(ptr[i])
            v_end = int(ptr[i + 1])
            x_padded[i, :V] = x_flat[v_start:v_end]
            f_start = face_ptr[i]
            f_end = face_ptr[i + 1]
            if F > 0:
                faces_padded[i, :F] = faces_flat[f_start:f_end]
            mass_padded[i, :V] = poisson_lists["poisson_vert_mass"][i][0, :V]
            M_padded[i, :2 * F] = poisson_lists["poisson_M"][i][0, :2 * F]
            G_src = poisson_lists["poisson_G"][i][0, :2 * F, :V]
            G_padded[i, :2 * F, :V] = G_src.to_dense() if G_src.is_sparse else G_src
            solver_list.append(self._build_solver(poisson_lists["poisson_L"][i], device))

        x_out, _ = self.block(
            x_padded, M_padded, G_padded, solver_list, faces_padded, mass_padded
        )

        for i in range(B):
            V = vert_counts[i]
            v_start = int(ptr[i])
            v_end = int(ptr[i + 1])
            x_flat[v_start:v_end] = x_out[i, :V]

        surface.x = x_flat
        return surface
