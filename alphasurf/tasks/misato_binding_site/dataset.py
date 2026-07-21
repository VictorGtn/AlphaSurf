"""Dataset and geometry construction for MISATO binding-site prediction."""

from __future__ import annotations

import logging
import os
from typing import Optional

import h5py
import numpy as np
import torch
from alphasurf.protein.features import Features
from alphasurf.protein.graphs import atom_coords_to_edges, res_type_to_hphob
from alphasurf.protein.residue_graph import ResidueGraph
from alphasurf.protein.surfaces import SurfaceObject
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class MisatoBindingSiteDataset(Dataset):
    """One protein per sample with a binary target for every residue."""

    def __init__(
        self,
        pdb_ids,
        data_dir,
        md_path,
        surface_cfg,
        graph_cfg,
        frame_mode="first",
        frame_index=0,
        frame_fraction=0.5,
        noise_sigma=0.0,
    ):
        self.pdb_ids = list(pdb_ids)
        self.data_dir = data_dir
        self.md_path = md_path
        self.surface_cfg = surface_cfg
        self.graph_cfg = graph_cfg
        self.frame_mode = frame_mode
        self.frame_index = int(frame_index)
        self.frame_fraction = float(frame_fraction)
        self.noise_sigma = float(noise_sigma)
        self._md = None
        self._md_pid = None

    def __len__(self):
        return len(self.pdb_ids)

    def __getstate__(self):
        state = self.__dict__.copy()
        # h5py handles cannot be shared safely across DataLoader processes.
        state["_md"] = None
        state["_md_pid"] = None
        return state

    def _get_md(self):
        pid = os.getpid()
        if self._md is None or self._md_pid != pid:
            if self._md is not None:
                self._md.close()
            self._md = h5py.File(self.md_path, "r")
            self._md_pid = pid
        return self._md

    @staticmethod
    def _lookup_group(md, pdb_id):
        for key in (pdb_id, pdb_id.lower(), pdb_id.upper()):
            if key in md:
                return md[key]
        raise KeyError(f"{pdb_id} is absent from MD.hdf5")

    def _load_frame(self, pdb_id, item):
        group = self._lookup_group(self._get_md(), pdb_id)
        if "trajectory_coordinates" not in group:
            raise KeyError(f"{pdb_id} has no trajectory_coordinates")
        trajectory = group["trajectory_coordinates"]
        n_frames = trajectory.shape[0]
        if self.frame_mode == "random":
            frame_idx = int(torch.randint(n_frames, (1,)).item())
        elif self.frame_mode in {"first", "fixed"}:
            frame_idx = self.frame_index
        elif self.frame_mode == "middle":
            frame_idx = n_frames // 2
        elif self.frame_mode == "fraction":
            if not 0.0 <= self.frame_fraction <= 1.0:
                raise ValueError(
                    f"frame_fraction must be in [0, 1], got {self.frame_fraction}"
                )
            # Round halves upward so fraction=0.5 agrees with n_frames // 2.
            frame_idx = int(self.frame_fraction * (n_frames - 1) + 0.5)
        else:
            raise ValueError(f"Unsupported frame_mode: {self.frame_mode}")
        if not 0 <= frame_idx < n_frames:
            raise IndexError(f"Frame {frame_idx} outside [0, {n_frames}) for {pdb_id}")
        if "protein_source_index" in item:
            source_index = item["protein_source_index"].numpy()
            atom_pos = torch.from_numpy(
                np.asarray(trajectory[frame_idx, source_index], dtype=np.float32)
            )
        else:
            # Backward compatibility for caches made from an already
            # hydrogen-stripped adaptability_MD.hdf5 file.
            ligand_start = int(item["ligand_start"])
            atom_pos = torch.from_numpy(
                np.asarray(trajectory[frame_idx, :ligand_start], dtype=np.float32)
            )
        ca_index = item["ca_atom_index"].long()
        return atom_pos, atom_pos[ca_index], frame_idx

    def __getitem__(self, idx) -> Optional[Data]:
        pdb_id = self.pdb_ids[idx]
        path = os.path.join(self.data_dir, f"{pdb_id.lower()}.pt")
        if not os.path.exists(path):
            logger.warning("Preprocessed sample missing: %s", path)
            return None
        try:
            item = torch.load(path, weights_only=False, map_location="cpu")
            atom_pos, ca_pos, frame_idx = self._load_frame(pdb_id, item)
            if self.noise_sigma > 0:
                # A shared random translation matches the MISATO baseline protocol.
                shift = torch.randn(1, 3) * self.noise_sigma
                atom_pos = atom_pos + shift
                ca_pos = ca_pos + shift
            graph = self._build_graph(item, ca_pos)
            surface = self._build_surface(atom_pos, item["atom_radius"].float(), pdb_id)
            if graph is None or surface is None or len(item["y"]) != graph.num_res:
                return None
            return Data(
                graph=graph,
                surface=surface,
                y=item["y"].long(),
                pdb_id=pdb_id,
                frame_idx=frame_idx,
            )
        except Exception as exc:
            logger.warning("Failed to build %s: %s", pdb_id, exc)
            return None

    def _build_graph(self, item, ca_pos):
        cutoff = float(getattr(self.graph_cfg, "edge_cutoff", 12.0))
        edge_index, edge_attr = atom_coords_to_edges(ca_pos, edge_dist_cutoff=cutoff)
        residue_type = item["residue_type"].long()
        graph = ResidueGraph(
            node_pos=ca_pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_names=torch.arange(len(ca_pos)),
        )
        graph.features = Features(num_nodes=len(ca_pos))
        graph.features.add_named_oh_features("amino_types", residue_type, 21)
        hphob = torch.tensor(
            [res_type_to_hphob[int(code)] for code in residue_type],
            dtype=torch.float32,
        )
        graph.features.add_named_features("hphobs", hphob)
        graph.node_len = len(ca_pos)
        graph.expand_features(
            remove_feats=True,
            feature_keys=getattr(self.graph_cfg, "feat_keys", "all"),
            oh_keys=getattr(self.graph_cfg, "oh_keys", "all"),
        )
        return graph

    def _build_surface(self, atom_pos, atom_radius, pdb_id):
        from alphasurf.protein.create_surface import pdb_to_alpha_complex

        cfg = self.surface_cfg
        verts, faces = pdb_to_alpha_complex(
            pdb_path=f"{pdb_id}.pdb",
            alpha_value=float(getattr(cfg, "alpha_value", 0.0)),
            atom_pos=atom_pos.numpy().astype(np.float32),
            atom_radius=atom_radius.numpy().astype(np.float32),
        )
        surface = SurfaceObject.from_verts_faces(
            verts=verts,
            faces=faces,
            face_reduction_rate=float(getattr(cfg, "face_reduction_rate", 1.0)),
            max_vert_number=int(getattr(cfg, "max_vert_number", 100000)),
            min_vert_number=int(getattr(cfg, "min_vert_number", 16)),
            use_pymesh=bool(getattr(cfg, "use_pymesh", False)),
            surface_method="alpha_complex",
            obj_name=pdb_id,
            use_igl_normals=bool(getattr(cfg, "use_igl_normals", False)),
            use_poisson=bool(getattr(cfg, "use_poisson", False)),
            poisson_high_precision=bool(getattr(cfg, "poisson_high_precision", True)),
            tufting=bool(getattr(cfg, "tufting", True)),
        )
        surface.add_geom_feats()
        surface.from_numpy()
        surface.expand_features(
            remove_feats=True,
            feature_keys=getattr(cfg, "feat_keys", "all"),
            oh_keys=getattr(cfg, "oh_keys", "all"),
        )
        return surface


def load_ids(path):
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]
