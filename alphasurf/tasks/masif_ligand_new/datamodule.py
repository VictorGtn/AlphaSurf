"""
MaSIF-Ligand data module using the base infrastructure.
"""

import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.protein.transforms import NoiseAugmentor, PatchExtractor
from alphasurf.tasks.masif_ligand_new.dataset import (
    MasifLigandDataset,
    load_ligand_data,
)
from alphasurf.utils.data_utils import AtomBatch, update_model_input_dim


class MasifLigandDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for MaSIF-Ligand task.

    Supports both disk-based and on-the-fly modes with integrated noise augmentation.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Resolve data directories
        data_dir = cfg.data_dir
        raw_data_dir = os.path.join(data_dir, "raw_data_MasifLigand")
        splits_dir = os.path.join(raw_data_dir, "splits")
        ligands_dir = os.path.join(raw_data_dir, "ligand")
        pdb_dir = os.path.join(raw_data_dir, "pdb")
        patch_dir = os.path.join(data_dir, "dataset_MasifLigand")

        # Load pocket data for all splits
        self.pocket_data = {}
        for split in ["train", "val", "test"]:
            split_file = os.path.join(splits_dir, f"{split}-list.txt")
            cache_file = os.path.join(splits_dir, f"{split}.p")
            self.pocket_data[split] = load_ligand_data(
                split_file, ligands_dir, cache_file
            )

        # Build protein loader
        self.protein_loader = self._build_protein_loader(cfg, pdb_dir, patch_dir)

        # Build noise augmentor (only for training)
        self.noise_augmentor = self._build_noise_augmentor(cfg)

        # DataLoader arguments
        self.loader_args = self._build_loader_args(cfg)

        # Update model input dimensions
        temp_dataset = MasifLigandDataset(
            self.pocket_data["train"],
            self.protein_loader,
            apply_noise=False,
        )
        update_model_input_dim(cfg, dataset_temp=temp_dataset)

    def _build_protein_loader(self, cfg, pdb_dir: str, patch_dir: str) -> ProteinLoader:
        """Build ProteinLoader with appropriate configuration."""
        on_fly_cfg = getattr(cfg, "on_fly", None)

        # Determine mode: on_fly if we have on_fly config, else disk
        mode = "on_fly" if on_fly_cfg is not None else "disk"

        # Surface and graph directories for disk mode
        surface_dir = None
        graph_dir = None
        if mode == "disk":
            surface_dir = os.path.join(
                cfg.cfg_surface.data_dir, cfg.cfg_surface.data_name
            )
            graph_dir = os.path.join(cfg.cfg_graph.data_dir, cfg.cfg_graph.data_name)

        # ESM directory
        esm_dir = getattr(cfg.cfg_graph, "esm_dir", None)
        if on_fly_cfg is not None:
            esm_dir = getattr(on_fly_cfg, "esm_dir", esm_dir)

        # Build patch extractor for on-fly mode
        patch_extractor = None
        if mode == "on_fly":
            use_whole = getattr(on_fly_cfg, "use_whole_surfaces", True)
            if not use_whole and os.path.isdir(patch_dir):
                patch_extractor = PatchExtractor(
                    patch_dir=patch_dir,
                    radius=getattr(on_fly_cfg, "patch_radius", 6.0),
                    min_verts=getattr(on_fly_cfg, "min_verts", 140),
                    max_radius=getattr(on_fly_cfg, "patch_max_radius", 12.0),
                )

        # Build noise augmentor
        noise_augmentor = self._build_noise_augmentor(cfg)

        # Merge surface config with on_fly overrides
        surface_config = self._merge_surface_config(cfg, on_fly_cfg)
        graph_config = self._merge_graph_config(cfg, on_fly_cfg)

        return ProteinLoader(
            mode=mode,
            pdb_dir=pdb_dir,
            surface_dir=surface_dir,
            graph_dir=graph_dir,
            esm_dir=esm_dir,
            surface_config=surface_config,
            graph_config=graph_config,
            noise_augmentor=noise_augmentor,
            patch_extractor=patch_extractor,
        )

    def _build_noise_augmentor(self, cfg) -> Optional[NoiseAugmentor]:
        """Build NoiseAugmentor from config."""
        on_fly_cfg = getattr(cfg, "on_fly", None)
        if on_fly_cfg is None:
            return None

        noise_mode = getattr(on_fly_cfg, "noise_mode", "none")
        if noise_mode == "none":
            return None

        return NoiseAugmentor(
            mode=noise_mode,
            sigma_graph=getattr(on_fly_cfg, "sigma_graph", 0.3),
            sigma_mesh=getattr(on_fly_cfg, "sigma_mesh", 0.3),
            clip_sigma=getattr(on_fly_cfg, "clip_sigma", 3.0),
        )

    def _merge_surface_config(self, cfg, on_fly_cfg):
        """Merge cfg_surface with on_fly overrides."""
        from torch_geometric.data import Data

        merged = Data()

        # Copy base config
        for key in dir(cfg.cfg_surface):
            if not key.startswith("_"):
                try:
                    setattr(merged, key, getattr(cfg.cfg_surface, key))
                except Exception:
                    pass

        # Apply on_fly overrides
        if on_fly_cfg is not None:
            for key in [
                "surface_method",
                "alpha_value",
                "face_reduction_rate",
                "max_vert_number",
                "use_pymesh",
                "use_whole_surfaces",
                "precomputed_patches_dir",
            ]:
                if hasattr(on_fly_cfg, key):
                    setattr(merged, key, getattr(on_fly_cfg, key))

        return merged

    def _merge_graph_config(self, cfg, on_fly_cfg):
        """Merge cfg_graph with on_fly overrides."""
        from torch_geometric.data import Data

        merged = Data()

        for key in dir(cfg.cfg_graph):
            if not key.startswith("_"):
                try:
                    setattr(merged, key, getattr(cfg.cfg_graph, key))
                except Exception:
                    pass

        if on_fly_cfg is not None and hasattr(on_fly_cfg, "esm_dir"):
            merged.esm_dir = on_fly_cfg.esm_dir

        return merged

    def _build_loader_args(self, cfg) -> dict:
        """Build DataLoader arguments from config."""
        loader_cfg = cfg.loader
        args = {
            "num_workers": getattr(loader_cfg, "num_workers", 0),
            "batch_size": getattr(loader_cfg, "batch_size", 8),
            "pin_memory": getattr(loader_cfg, "pin_memory", False),
            "drop_last": getattr(loader_cfg, "drop_last", False),
            "collate_fn": self._collate_fn,
        }

        if args["num_workers"] > 0:
            prefetch = getattr(loader_cfg, "prefetch_factor", None)
            if prefetch is not None:
                args["prefetch_factor"] = prefetch
            args["persistent_workers"] = getattr(
                loader_cfg, "persistent_workers", False
            )

        return args

    @staticmethod
    def _collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return AtomBatch.from_data_list(batch)

    def train_dataloader(self) -> DataLoader:
        dataset = MasifLigandDataset(
            self.pocket_data["train"],
            self.protein_loader,
            apply_noise=True,
        )
        shuffle = getattr(self.cfg.loader, "shuffle", True)
        return DataLoader(dataset, shuffle=shuffle, **self.loader_args)

    def val_dataloader(self) -> DataLoader:
        dataset = MasifLigandDataset(
            self.pocket_data["val"],
            self.protein_loader,
            apply_noise=False,
        )
        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self) -> DataLoader:
        dataset = MasifLigandDataset(
            self.pocket_data["test"],
            self.protein_loader,
            apply_noise=False,
        )
        return DataLoader(dataset, shuffle=False, **self.loader_args)
