"""
Lightning DataModule for CATH S3F-style pretraining.

Three data paths:
  - s3f_exact encoder, `on_fly` null: loads precomputed dMaSIF point clouds
    from `precompute_dir` via CATHDatasetS3FExact.
  - s3f_exact encoder, `on_fly` set: parses PDBs via CATHDatasetS3FExactOnFly.
    With `on_fly.surface_in_workers` the dMaSIF cloud is generated per protein
    on CPU inside the dataloader worker; otherwise it is generated for each
    batch in `on_after_batch_transfer`, once the batch is on the GPU.
  - other encoders: on-the-fly ProteinLoader pipeline (alpha-complex mesh
    + residue graph) via CATHDataset.

For non-s3f_exact encoders, the model forward concatenates the 31-dim graph
features with the 1280-dim ESM embedding (31 + 1280 = 1311).
"""

from __future__ import annotations


import pytorch_lightning as pl
from alphasurf.utils.data_utils import AtomBatch
from torch.utils.data import DataLoader


class S3FPretrainDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_name = getattr(cfg.encoder, "name", "")
        self.is_s3f_exact = "s3f_exact" in self.encoder_name

        # For s3f_exact, `on_fly` selects runtime dMaSIF generation; leaving it
        # null uses `precompute_dir`, as elsewhere in the repo.
        self.s3f_on_fly = self.is_s3f_exact and getattr(cfg, "on_fly", None) is not None
        self.s3f_surface_in_workers = self.s3f_on_fly and bool(
            getattr(cfg.on_fly, "surface_in_workers", True)
        )

        if self.s3f_on_fly:
            self.pdb_dir = cfg.data_dir
        elif self.is_s3f_exact:
            if not getattr(cfg, "precompute_dir", None):
                raise ValueError(
                    "precompute_dir must be set for s3f_exact encoder "
                    "(run precompute_s3f_exact.py first, or set on_fly)"
                )
            self.precompute_dir = cfg.precompute_dir
        else:
            self._setup_on_fly(cfg)

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

        if not self.is_s3f_exact:
            self._update_encoder_input_dim(cfg)

    def _setup_on_fly(self, cfg):
        from alphasurf.protein.protein_loader import ProteinLoader
        from alphasurf.utils.config_utils import merge_surface_config

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

    def _update_encoder_input_dim(self, cfg):
        from alphasurf.tasks.s3f_pretrain.dataset import CATHDataset
        from alphasurf.tasks.s3f_pretrain.model import ESM_EMBED_DIM
        from alphasurf.utils.data_utils import update_model_input_dim
        from omegaconf import open_dict

        temp_dataset = CATHDataset(
            pdb_dir=self.pdb_dir,
            split="train",
            protein_loader=self.protein_loader,
            mask_rate=getattr(cfg, "mask_rate", 0.15),
            max_length=getattr(cfg, "max_length", 250),
            k_surf_leak=getattr(cfg, "k_surf_leak", 20),
            structure_mask_mode=self._structure_mask_mode(cfg),
        )
        update_model_input_dim(cfg, temp_dataset, gkey="graph", skey="surface")

        with open_dict(cfg):
            block0 = cfg.encoder.blocks[0]
            if "g_pre_block" in block0:
                block0.g_pre_block.dim_in = 31 + ESM_EMBED_DIM
            surface_esm_cfg = getattr(cfg, "surface_esm", None)
            surface_esm_enabled = bool(
                surface_esm_cfg is not None
                and getattr(surface_esm_cfg, "enabled", False)
            )
            if surface_esm_enabled:
                if "s_pre_block" not in block0:
                    raise ValueError(
                        "surface_esm requires an s_pre_block in the first encoder block"
                    )
                surface_esm_mode = str(getattr(surface_esm_cfg, "mode", "light"))
                if surface_esm_mode == "light":
                    block0.s_pre_block.dim_in += cfg.cfg_head.encoded_dims
                elif surface_esm_mode == "s3f_full":
                    surface_esm_cfg.input_dim = block0.s_pre_block.dim_in
                    block0.s_pre_block.dim_in = ESM_EMBED_DIM
                else:
                    raise ValueError(f"Unknown surface_esm.mode: {surface_esm_mode}")

    @staticmethod
    def _collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return AtomBatch.from_data_list(batch)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Generate the batch's dMaSIF surfaces once it is on the GPU."""
        if not self.s3f_on_fly or self.s3f_surface_in_workers or batch is None:
            return batch
        from alphasurf.tasks.s3f_pretrain.dataset_s3f_exact import attach_surfaces

        return attach_surfaces(batch, device=batch.graph.node_pos.device)

    def _create_dataset(self, split: str):
        seed = getattr(self.cfg, "seed", 0) + (0 if split == "train" else 1)
        if self.s3f_on_fly:
            from alphasurf.tasks.s3f_pretrain.dataset_s3f_exact import (
                CATHDatasetS3FExactOnFly,
            )

            return CATHDatasetS3FExactOnFly(
                pdb_dir=self.pdb_dir,
                split=split,
                mask_rate=getattr(self.cfg, "mask_rate", 0.15),
                max_length=getattr(self.cfg, "max_length", 250),
                seed=seed,
                surface_in_workers=self.s3f_surface_in_workers,
            )

        if self.is_s3f_exact:
            from alphasurf.tasks.s3f_pretrain.dataset_s3f_exact import (
                CATHDatasetS3FExact,
            )

            return CATHDatasetS3FExact(
                precompute_dir=self.precompute_dir,
                split=split,
                mask_rate=getattr(self.cfg, "mask_rate", 0.15),
                max_length=getattr(self.cfg, "max_length", 250),
                seed=seed,
            )

        from alphasurf.tasks.s3f_pretrain.dataset import CATHDataset

        return CATHDataset(
            pdb_dir=self.pdb_dir,
            split=split,
            protein_loader=self.protein_loader,
            mask_rate=getattr(self.cfg, "mask_rate", 0.15),
            max_length=getattr(self.cfg, "max_length", 250),
            k_surf_leak=getattr(self.cfg, "k_surf_leak", 20),
            structure_mask_mode=self._structure_mask_mode(self.cfg),
            seed=seed,
        )

    @staticmethod
    def _structure_mask_mode(cfg) -> str:
        mask_cfg = getattr(cfg, "structure_mask", None)
        return str(getattr(mask_cfg, "mode", "backbone"))

    def train_dataloader(self) -> DataLoader:
        dataset = self._create_dataset("train")
        return DataLoader(dataset, shuffle=True, **self.loader_args)

    def val_dataloader(self) -> DataLoader:
        dataset = self._create_dataset("val")
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self) -> DataLoader:
        dataset = self._create_dataset("test")
        return DataLoader(dataset, shuffle=False, **self.loader_args)
