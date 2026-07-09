"""
Lightning DataModule for CATH S3F-style pretraining.

Builds train/val/test CATHDataset splits, configures ProteinLoader for
on-the-fly alpha-complex surface + residue graph generation, and overrides
the encoder's graph input dim to account for the 1280-dim ESM embedding
concatenated in the model forward (31 + 1280 = 1311).
"""

from __future__ import annotations


import pytorch_lightning as pl
from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.s3f_pretrain.dataset import CATHDataset
from alphasurf.utils.config_utils import merge_surface_config
from alphasurf.utils.data_utils import AtomBatch
from omegaconf import open_dict
from torch.utils.data import DataLoader


class S3FPretrainDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pdb_dir = cfg.data_dir

        on_fly_cfg = getattr(cfg, "on_fly", None)
        surface_config = merge_surface_config(cfg.cfg_surface, on_fly_cfg)
        graph_config = merge_surface_config(cfg.cfg_graph, on_fly_cfg)
        surface_config.use_poisson = "poisson" in cfg.encoder.name

        self.protein_loader = ProteinLoader(
            mode="on_fly",
            pdb_dir=self.pdb_dir,
            surface_config=surface_config,
            graph_config=graph_config,
        )

        self.loader_args = {
            "num_workers": getattr(cfg.loader, "num_workers", 8),
            "batch_size": getattr(cfg.loader, "batch_size", 8),
            "pin_memory": getattr(cfg.loader, "pin_memory", False),
            "collate_fn": self._collate_fn,
        }
        if self.loader_args["num_workers"] > 0:
            self.loader_args["prefetch_factor"] = getattr(
                cfg.loader, "prefetch_factor", 2
            )
            self.loader_args["persistent_workers"] = getattr(
                cfg.loader, "persistent_workers", True
            )

        self._update_encoder_input_dim(cfg)

    def _update_encoder_input_dim(self, cfg):
        """Auto-infer surface dim via update_model_input_dim, then override
        graph dim to 31 + ESM_EMBED_DIM (1280) = 1311.

        The model forward concatenates ESM embeddings into graph.x before the
        encoder runs, so the graph input dim is 1311, not the 31 that
        update_model_input_dim would infer from the dataset's graph.x.
        """
        from alphasurf.tasks.s3f_pretrain.model import ESM_EMBED_DIM
        from alphasurf.utils.data_utils import update_model_input_dim

        temp_dataset = CATHDataset(
            pdb_dir=self.pdb_dir,
            split="train",
            protein_loader=self.protein_loader,
            mask_rate=getattr(cfg, "mask_rate", 0.15),
            max_length=getattr(cfg, "max_length", 250),
            k_surf_leak=getattr(cfg, "k_surf_leak", 20),
        )
        update_model_input_dim(cfg, temp_dataset, gkey="graph", skey="surface")

        with open_dict(cfg):
            block0 = cfg.encoder.blocks[0]
            if "g_pre_block" in block0:
                block0.g_pre_block.dim_in = 31 + ESM_EMBED_DIM

    @staticmethod
    def _collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return AtomBatch.from_data_list(batch)

    def _create_dataset(self, split: str) -> CATHDataset:
        return CATHDataset(
            pdb_dir=self.pdb_dir,
            split=split,
            protein_loader=self.protein_loader,
            mask_rate=getattr(self.cfg, "mask_rate", 0.15),
            max_length=getattr(self.cfg, "max_length", 250),
            k_surf_leak=getattr(self.cfg, "k_surf_leak", 20),
            seed=getattr(self.cfg, "seed", 0) + (0 if split == "train" else 1),
        )

    def train_dataloader(self) -> DataLoader:
        dataset = self._create_dataset("train")
        return DataLoader(dataset, shuffle=True, **self.loader_args)

    def val_dataloader(self) -> DataLoader:
        dataset = self._create_dataset("val")
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self) -> DataLoader:
        dataset = self._create_dataset("test")
        return DataLoader(dataset, shuffle=False, **self.loader_args)
