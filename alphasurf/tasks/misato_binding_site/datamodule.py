"""Lightning DataModule for the MISATO residue binding-site task."""

from __future__ import annotations

import os

import pytorch_lightning as pl
from alphasurf.tasks.misato_binding_site.dataset import (
    MisatoBindingSiteDataset,
    load_ids,
)
from alphasurf.utils.data_utils import AtomBatch, update_model_input_dim
from torch.utils.data import DataLoader


class MisatoBindingSiteDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.sample_dir = os.path.join(cfg.data_dir, "binding_site")
        self.md_path = getattr(cfg, "md_path", os.path.join(cfg.data_dir, "MD.hdf5"))
        self.split_dir = getattr(cfg, "split_dir", os.path.join(cfg.data_dir, "splits"))
        self.ids = {
            split: load_ids(os.path.join(self.split_dir, f"{split}.txt"))
            for split in ("train", "val", "test")
        }
        self._check_inputs()
        self.loader_args = self._loader_args()
        update_model_input_dim(cfg, self._dataset("train", for_probe=True))

    def _check_inputs(self):
        if not os.path.isdir(self.sample_dir):
            raise FileNotFoundError(
                f"Missing {self.sample_dir}; run misato_binding_site/preprocess.sh first"
            )
        if not os.path.isfile(self.md_path):
            raise FileNotFoundError(f"Missing MISATO trajectory file: {self.md_path}")
        counts = {
            split: sum(
                os.path.exists(os.path.join(self.sample_dir, f"{pid.lower()}.pt"))
                for pid in ids
            )
            for split, ids in self.ids.items()
        }
        if counts["train"] == 0:
            raise RuntimeError("No preprocessed training samples match the split files")

    def _loader_args(self):
        loader = self.cfg.loader
        args = {
            "batch_size": getattr(loader, "batch_size", 8),
            "num_workers": getattr(loader, "num_workers", 8),
            "pin_memory": getattr(loader, "pin_memory", False),
            "collate_fn": self._collate,
        }
        if args["num_workers"] > 0:
            args["persistent_workers"] = getattr(loader, "persistent_workers", True)
            prefetch = getattr(loader, "prefetch_factor", None)
            if prefetch is not None:
                args["prefetch_factor"] = prefetch
        return args

    def _dataset(self, split, for_probe=False):
        ids = self.ids[split]
        if for_probe:
            ids = ids[:50]
        noise_sigma = 0.0
        frame_mode = getattr(self.cfg, "eval_frame_mode", "first")
        frame_index = getattr(self.cfg, "eval_frame_index", 0)
        frame_fraction = getattr(self.cfg, "eval_frame_fraction", 0.5)
        if split == "train":
            noise_sigma = float(getattr(self.cfg, "noise_sigma", 0.0))
            frame_mode = getattr(self.cfg, "train_frame_mode", "random")
            frame_index = getattr(
                self.cfg,
                "train_frame_index",
                getattr(self.cfg, "eval_frame_index", 0),
            )
            frame_fraction = getattr(
                self.cfg,
                "train_frame_fraction",
                getattr(self.cfg, "eval_frame_fraction", 0.5),
            )
        return MisatoBindingSiteDataset(
            pdb_ids=ids,
            data_dir=self.sample_dir,
            md_path=self.md_path,
            surface_cfg=self.cfg.cfg_surface,
            graph_cfg=self.cfg.cfg_graph,
            frame_mode=frame_mode,
            frame_index=frame_index,
            frame_fraction=frame_fraction,
            noise_sigma=noise_sigma,
        )

    @staticmethod
    def _collate(items):
        items = [item for item in items if item is not None]
        return AtomBatch.from_data_list(items) if items else None

    def train_dataloader(self):
        return DataLoader(
            self._dataset("train"),
            shuffle=getattr(self.cfg.loader, "shuffle", True),
            **self.loader_args,
        )

    def val_dataloader(self):
        return DataLoader(self._dataset("val"), shuffle=False, **self.loader_args)

    def test_dataloader(self):
        return DataLoader(self._dataset("test"), shuffle=False, **self.loader_args)
