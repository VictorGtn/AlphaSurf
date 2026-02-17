import os
import sys

import igl
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, "..", ".."))

import alphasurf.utils.torch_utils as diff_utils
from alphasurf.protein.features import Features, FeaturesHolder
from alphasurf.utils.python_utils import makedirs_path


def compute_HKS(evecs, evals, num_t, t_min=0.1, t_max=1000, scale=1000):
    evals = evals.flatten()
    assert evals[1] > 0
    assert np.min(evals) > -1e-6
    assert np.array_equal(evals, sorted(evals))

    t_list = np.geomspace(t_min, t_max, num_t, dtype=np.float32)
    phase = np.exp(-np.outer(t_list, evals[1:]))
    wphi = phase[:, None, :] * evecs[None, :, 1:]
    hks = np.einsum("tnk,nk->nt", wphi, evecs[:, 1:]) * scale
    heat_trace = np.sum(phase, axis=1)
    hks /= heat_trace
    return hks


def get_geom_feats(verts, faces, evecs, evals, vnormals, num_signatures=16):
    if len(faces) == 0 or len(verts) == 0:
        return None

    try:
        import igl

        _, _, k1, k2 = igl.principal_curvature(verts, faces)
    except:
        return None

    gauss_curvs = (k1 * k2).reshape(-1, 1)
    mean_curvs = (0.5 * (k1 + k2)).reshape(-1, 1)

    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)
    si = si.reshape(-1, 1)

    hks = compute_HKS(evecs, evals, num_signatures)
    geom_feats = np.concatenate([gauss_curvs, mean_curvs, si, hks, vnormals], axis=-1)
    return geom_feats


class SurfaceObject(Data, FeaturesHolder):
    def __init__(
        self,
        features=None,
        verts=None,
        faces=None,
        mass=None,
        L=None,
        evals=None,
        evecs=None,
        gradX=None,
        gradY=None,
        vnormals=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.verts = verts
        self.n_verts = len(self.verts) if verts is not None else 0
        self.faces = faces

        self.mass = mass
        self.L = L
        self.evals = evals
        self.evecs = evecs
        self.gradX = gradX
        self.gradY = gradY
        self.k_eig = len(evals) if evals is not None else 0
        self.set_vnormals(vnormals)

        if features is None:
            self.features = Features(num_nodes=self.n_verts)
        else:
            self.features = features

    def set_vnormals(self, vnormals=None, weighting="uniform", force=False):
        # Case 1: Directly set vnormals from the argument
        if vnormals is not None:
            self.vnormals = vnormals
            return

        # Case 2: Decide whether to compute/recompute normals
        # We compute if `force` is True, or if vnormals don't exist yet.
        should_compute = force or not (
            "vnormals" in self.keys() and self.vnormals is not None
        )

        if should_compute and "verts" in self.keys() and self.verts is not None:
            # Define weighting schemes
            weighting_map = {
                "uniform": igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM,
                "area": igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
                "angle": igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,
            }
            if weighting not in weighting_map:
                raise ValueError(
                    f"Unknown weighting type: {weighting}. Available options are {list(weighting_map.keys())}"
                )

            # Compute normals with the specified weighting
            computed_vnormals = igl.per_vertex_normals(
                diff_utils.toNP(self.verts),
                diff_utils.toNP(self.faces),
                weighting_map[weighting],
            )

            # Assign the computed normals, converting to torch tensor if necessary
            self.vnormals = (
                computed_vnormals
                if isinstance(self.verts, np.ndarray)
                else diff_utils.safe_to_torch(computed_vnormals)
            )

        # Case 3: Cannot compute (e.g., no verts), and not already present
        elif "vnormals" not in self.keys() or self.vnormals is None:
            self.vnormals = None

    def from_numpy(self, device="cpu"):
        for attr_name in ["verts", "faces", "evals", "evecs", "vnormals"]:
            attr_value = getattr(self, attr_name)
            setattr(
                self, attr_name, diff_utils.safe_to_torch(attr_value).to(device=device)
            )

        for attr_name in ["L", "mass", "gradX", "gradY"]:
            attr_value = getattr(self, attr_name)
            setattr(
                self,
                attr_name,
                diff_utils.sparse_np_to_pyg(attr_value).to(device=device),
            )
        return self

    def numpy(self):
        for attr_name in ["verts", "faces", "evals", "evecs", "vnormals"]:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.toNP(attr_value))

        for attr_name in ["L", "mass", "gradX", "gradY"]:
            attr_value = getattr(self, attr_name)
            setattr(self, attr_name, diff_utils.sparse_pyg_to_np(attr_value))
        return self

    def save(self, npz_path):
        self.numpy()
        np.savez(
            npz_path,
            verts=self.verts,
            faces=self.faces,
            mass_data=self.mass.data,
            mass_indices=self.mass.indices,
            mass_indptr=self.mass.indptr,
            mass_shape=self.mass.shape,
            L_data=self.L.data,
            L_indices=self.L.indices,
            L_indptr=self.L.indptr,
            L_shape=self.L.shape,
            evals=self.evals,
            evecs=self.evecs,
            vnormals=self.vnormals,
            gradX_data=self.gradX.data,
            gradX_indices=self.gradX.indices,
            gradX_indptr=self.gradX.indptr,
            gradX_shape=self.gradX.shape,
            gradY_data=self.gradY.data,
            gradY_indices=self.gradY.indices,
            gradY_indptr=self.gradY.indptr,
            gradY_shape=self.gradY.shape,
        )

    def save_torch(self, torch_path):
        self.from_numpy()
        makedirs_path(torch_path)
        torch.save(self, open(torch_path, "wb"))

    @classmethod
    def load(cls, npz_path):
        from alphasurf.protein.create_operators import load_operators

        npz_file = np.load(npz_path, allow_pickle=True)
        verts = npz_file["verts"]
        faces = npz_file["faces"]
        mass, L, evals, evecs, vnormals, gradX, gradY = load_operators(npz_file)

        return cls(
            verts=verts,
            faces=faces,
            mass=mass,
            L=L,
            evals=evals,
            evecs=evecs,
            vnormals=vnormals,
            gradX=gradX,
            gradY=gradY,
        )

    def add_geom_feats(self):
        from alphasurf.utils.timing_stats import Timer

        self.numpy()
        with Timer("geom_feats"):
            geom_feats = get_geom_feats(
                self.verts,
                self.faces,
                self.evecs,
                self.evals,
                self.vnormals,
            )

        if geom_feats is None:
            raise RuntimeError("Curvature calculation failed for current protein.")

        self.features.add_named_features("geom_feats", geom_feats)

    @classmethod
    def from_verts_faces(
        cls,
        verts,
        faces,
        min_vert_number=140,
        max_vert_number=50000,
        face_reduction_rate=1.0,
        use_fem_decomp=False,
        use_robust_laplacian=False,
        use_pymesh=False,
        out_ply_path=None,
        surface_method="msms",
        obj_name=None,
    ):
        from alphasurf.protein.create_operators import compute_operators
        from alphasurf.protein.create_surface import mesh_simplification
        from alphasurf.utils.timing_stats import Timer

        verts = diff_utils.toNP(verts)
        faces = diff_utils.toNP(faces).astype(int)
        with Timer("mesh_processing"):
            verts, faces, drop_ratio = mesh_simplification(
                verts=verts,
                faces=faces,
                out_ply=out_ply_path,
                face_reduction_rate=face_reduction_rate,
                min_vert_number=min_vert_number,
                max_vert_number=max_vert_number,
                use_pymesh=use_pymesh,
                surface_method=surface_method,
                obj_name=obj_name,
            )
        vnormals = igl.per_vertex_normals(verts, faces)

        with Timer("compute_operators"):
            frames, massvec, L, evals, evecs, gradX, gradY = compute_operators(
                verts,
                faces,
                normals=vnormals,
                use_fem_decomp=use_fem_decomp,
                use_robust_laplacian=use_robust_laplacian,
            )

        surface = cls(
            verts=verts,
            faces=faces,
            mass=massvec,
            L=L,
            evals=evals,
            evecs=evecs,
            gradX=gradX,
            gradY=gradY,
            vnormals=vnormals,
        )
        surface.drop_ratio = drop_ratio
        return surface

    @classmethod
    def from_pdb_path(cls, pdb_path, obj_name=None, **kwargs):
        """

        :param pdb_path:
        :param kwargs: see arguments for from_verts_faces. Also accepts:
            - atom_pos: Pre-parsed atom positions (for alpha_complex method)
            - atom_radius: Pre-parsed atom radii (for alpha_complex method)
        :return:
        """

        surface_method = kwargs.pop("surface_method", "msms")
        sbl_exe_path = kwargs.pop("sbl_exe_path", None)
        alpha_value = kwargs.pop("alpha_value", 0.1)
        atom_pos = kwargs.pop("atom_pos", None)
        atom_radius = kwargs.pop("atom_radius", None)

        from alphasurf.utils.timing_stats import Timer

        with Timer("surface_generation"):
            if surface_method == "msms":
                from alphasurf.protein.create_surface import pdb_to_surf_with_min

                verts, faces = pdb_to_surf_with_min(pdb_path)
            elif surface_method == "alpha_complex":
                from alphasurf.protein.create_surface import pdb_to_alpha_complex

                verts, faces, _, _ = pdb_to_alpha_complex(
                    pdb_path,
                    sbl_exe_path=sbl_exe_path,
                    alpha_value=alpha_value,
                    atom_pos=atom_pos,
                    atom_radius=atom_radius,
                )
            else:
                raise ValueError(f"Unknown surface method: {surface_method}")
        return cls.from_verts_faces(
            verts, faces, surface_method=surface_method, obj_name=obj_name, **kwargs
        )

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["mass", "L", "gradX", "gradY"]:
            return (0, 1)
        else:
            return Data.__cat_dim__(None, key, value, *args, **kwargs)


class SurfaceBatch(Batch):
    """
    This class is useful for PyG Batching

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def batch_from_data_list(cls, data_list):
        # Pad evecs/evals to match the maximum k_eig in the batch
        # This is needed for small meshes where k_eig < default (128)
        max_k_eig = 0
        for surface in data_list:
            if hasattr(surface, "evecs") and surface.evecs is not None:
                max_k_eig = max(max_k_eig, surface.evecs.shape[1])

        for surface in data_list:
            # Pad evecs (V, k) -> (V, max_k)
            if hasattr(surface, "evecs") and surface.evecs is not None:
                k = surface.evecs.shape[1]
                if k < max_k_eig:
                    pad_size = max_k_eig - k
                    if torch.is_tensor(surface.evecs):
                        surface.evecs = torch.nn.functional.pad(
                            surface.evecs, (0, pad_size)
                        )
                    else:
                        import numpy as np

                        surface.evecs = np.pad(surface.evecs, ((0, 0), (0, pad_size)))

            # Pad evals (k,) -> (max_k,)
            if hasattr(surface, "evals") and surface.evals is not None:
                k = surface.evals.shape[0]
                if k < max_k_eig:
                    pad_size = max_k_eig - k
                    if torch.is_tensor(surface.evals):
                        surface.evals = torch.nn.functional.pad(
                            surface.evals, (0, pad_size)
                        )
                    else:
                        import numpy as np

                        surface.evals = np.pad(surface.evals, (0, pad_size))

            # Convert sparse tensors to PyG format
            # This is needed for data that was created as torch sparse tensor (instead of pyg ones),
            # since they cannot be batched
            # Note: dmasif surfaces (point clouds) don't have spectral operators like mass, L, etc.
            for key in {"L", "mass", "gradX", "gradY"}:
                if hasattr(surface, key):
                    tensor_coo = getattr(surface, key)
                    if isinstance(tensor_coo, torch.Tensor):
                        pyg_tensor = SparseTensor.from_torch_sparse_coo_tensor(
                            tensor_coo
                        )
                    else:
                        pyg_tensor = tensor_coo
                    surface[key] = pyg_tensor
        batch = Batch.from_data_list(data_list)
        batch = batch.contiguous()
        surface_batch = cls()
        surface_batch.__dict__.update(batch.__dict__)
        return surface_batch

    def __cat_dim__(self, key, value, *args, **kwargs):
        return SurfaceObject.__cat_dim__(None, key, value, *args, **kwargs)

    def to_lists(self):
        if "cache" not in self.keys():
            surfaces = self.to_data_list()
            x_in = [mini_surface.x for mini_surface in surfaces]
            mass = [mini_surface.mass for mini_surface in surfaces]
            L = [mini_surface.L for mini_surface in surfaces]
            evals = [mini_surface.evals for mini_surface in surfaces]
            evecs = [mini_surface.evecs for mini_surface in surfaces]
            vnormals = [mini_surface.vnormals for mini_surface in surfaces]
            gradX = [mini_surface.gradX for mini_surface in surfaces]
            gradY = [mini_surface.gradY for mini_surface in surfaces]
            self.cache = mass, L, evals, evecs, vnormals, gradX, gradY
        else:
            mass, L, evals, evecs, vnormals, gradX, gradY = self.cache
            surface_sizes = list(self.n_verts.detach().cpu().numpy())
            x_in = torch.split(self.x, surface_sizes)
        return x_in, mass, L, evals, evecs, gradX, gradY


if __name__ == "__main__":
    pass
    surface_file = "../../data/example_files/example_operator.npz"
    surface = SurfaceObject.load(surface_file)
    surface = surface.from_numpy()
    surface = surface.numpy()
    surface.add_geom_feats()

    # Save as np
    surface_file_np = "../../data/example_files/example_surface.npz"
    surface.save(surface_file_np)
    # Save as torch, a bit heavier
    surface_file_torch = "../../data/example_files/example_surface.pt"
    surface.save_torch(surface_file_torch)

    # t0 = time.time()
    # for _ in range(100):
    #     surface = SurfaceObject.load(surface_file_np)
    # print('np', time.time() - t0)
    #
    # t0 = time.time()
    # for _ in range(100):
    #     surface = torch.load(surface_file_torch)
    # print('torch', time.time() - t0)
    # torch is MUCH faster : 4.34 vs 0.7...

    verts, faces = surface.verts, surface.faces
    surface_hmr = SurfaceObject.from_verts_faces(verts, faces, use_fem_decomp=False)

    surface_1ycr_large = SurfaceObject.from_pdb_path(
        "../../data/example_files/1ycr_A.pdb",
        use_fem_decomp=False,
        face_reduction_rate=1.0,
    )
    surface_file_torch = "../../data/example_files/1ycr_large.pt"
    surface_1ycr_large.save_torch(surface_file_torch)

    surface_1ycr_small = SurfaceObject.from_pdb_path(
        "../../data/example_files/1ycr_A.pdb",
        use_fem_decomp=False,
        face_reduction_rate=0.1,
    )
    surface_file_torch = "../../data/example_files/1ycr_small.pt"
    surface_1ycr_small.save_torch(surface_file_torch)
    a = 1
