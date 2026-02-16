"""
PINDER-Pair data module.
"""

import os
from typing import Optional

import pytorch_lightning as pl
from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.protein.transforms import NoiseAugmentor
from alphasurf.tasks.pinder_pair.dataset import (
    PinderAlignedDataset,
    PinderPairDataset,
    load_pinder_split,
)
from alphasurf.utils.batch_sampler import AtomBudgetBatchSampler
from alphasurf.utils.data_utils import AtomBatch, update_model_input_dim
from torch.utils.data import DataLoader


class PinderPairDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for PINDER protein-protein interaction task.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Data directories
        data_dir = cfg.data_dir
        self.pdb_dir = os.path.join(data_dir, "pdb")

        # Test setting: 'holo' (bound), 'apo' (unbound), 'af2' (predicted)
        self.test_setting = getattr(cfg, "test_setting", "apo")

        # Load splits
        self.systems = {}
        for split in ["train", "val", "test"]:
            self.systems[split] = load_pinder_split(
                data_dir,
                split,
                test_setting=self.test_setting if split == "test" else None,
            )

        # Build protein loader (one shared loader for all splits)
        self.protein_loader = self._build_protein_loader(cfg)

        # Task-specific params
        self.neg_to_pos_ratio = getattr(cfg, "neg_to_pos_ratio", 1.0)
        self.surface_neg_to_pos_ratio = getattr(cfg, "surface_neg_to_pos_ratio", 10.0)
        self.max_pos_per_pair = getattr(cfg, "max_pos_per_pair", -1)
        # Interface distances
        self.interface_distance = getattr(cfg, "interface_distance", 8.0)
        self.interface_distance_graph = getattr(
            cfg, "interface_distance_graph", self.interface_distance
        )
        self.interface_distance_surface = getattr(
            cfg, "interface_distance_surface", self.interface_distance
        )

        # DataLoader args
        self.loader_args = self._build_loader_args(cfg)

        # Update model input dimensions
        temp_dataset = PinderPairDataset(
            systems=self.systems["train"][:10],
            protein_loader=self.protein_loader,
            pdb_dir=self.pdb_dir,
            apply_noise=False,
            interface_distance_graph=self.interface_distance_graph,
            interface_distance_surface=self.interface_distance_surface,
        )
        update_model_input_dim(
            cfg, dataset_temp=temp_dataset, gkey="graph_1", skey="surface_1"
        )

    def _build_protein_loader(self, cfg) -> ProteinLoader:
        """Build ProteinLoader with appropriate configuration."""
        on_fly_cfg = getattr(cfg, "on_fly", None)
        mode = "on_fly" if on_fly_cfg is not None else "disk"

        # Directories for disk mode
        surface_dir = None
        graph_dir = None
        if mode == "disk":
            surface_dir = os.path.join(
                cfg.cfg_surface.data_dir, cfg.cfg_surface.data_name
            )
            graph_dir = os.path.join(cfg.cfg_graph.data_dir, cfg.cfg_graph.data_name)

        esm_dir = getattr(cfg.cfg_graph, "esm_dir", None)

        # Noise augmentor
        noise_augmentor = self._build_noise_augmentor(cfg)

        # Merge configs
        surface_config = self._merge_config(cfg.cfg_surface, on_fly_cfg)
        graph_config = self._merge_config(cfg.cfg_graph, on_fly_cfg)

        return ProteinLoader(
            mode=mode,
            pdb_dir=self.pdb_dir,
            surface_dir=surface_dir,
            graph_dir=graph_dir,
            esm_dir=esm_dir,
            surface_config=surface_config,
            graph_config=graph_config,
            noise_augmentor=noise_augmentor,
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

    def _merge_config(self, base_cfg, override_cfg):
        """Merge base config with on_fly overrides."""
        from alphasurf.utils.config_utils import merge_surface_config

        return merge_surface_config(base_cfg, override_cfg)

    def _build_loader_args(self, cfg) -> dict:
        """Build DataLoader arguments."""
        loader_cfg = cfg.loader
        args = {
            "num_workers": getattr(loader_cfg, "num_workers", 0),
            "batch_size": getattr(loader_cfg, "batch_size", 8),
            "pin_memory": getattr(loader_cfg, "pin_memory", False),
            "collate_fn": self._collate_fn,
        }

        if args["num_workers"] > 0:
            prefetch = getattr(loader_cfg, "prefetch_factor", None)
            if prefetch is not None:
                args["prefetch_factor"] = prefetch
            args["persistent_workers"] = getattr(
                loader_cfg, "persistent_workers", False
            )
            # Use spawn instead of fork for CGAL/alpha_complex compatibility
            # args["multiprocessing_context"] = "spawn"

        return args

    @staticmethod
    def _collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return AtomBatch.from_data_list(batch)

    def _create_dataset(self, split: str, apply_noise: bool) -> PinderPairDataset:
        DatasetClass = PinderPairDataset
        # Use aligned dataset for Apo/AF2 testing to transfer labels from Holo
        if split == "test" and self.test_setting in ["apo", "af2"]:
            DatasetClass = PinderAlignedDataset

        return DatasetClass(
            systems=self.systems[split],
            protein_loader=self.protein_loader,
            pdb_dir=self.pdb_dir,
            apply_noise=apply_noise,
            neg_to_pos_ratio=self.neg_to_pos_ratio,
            max_pos_per_pair=self.max_pos_per_pair,
            interface_distance_graph=self.interface_distance_graph,
            interface_distance_surface=self.interface_distance_surface,
            surface_neg_to_pos_ratio=self.surface_neg_to_pos_ratio,
        )

    def train_dataloader(self) -> DataLoader:
        # Use dynamic batch sampler if configured
        use_dynamic_batching = getattr(self.cfg.loader, "use_dynamic_batching", False)

        if use_dynamic_batching:
            max_atoms = getattr(self.cfg.loader, "max_atoms_per_batch", 50000)
            min_batch_size = getattr(self.cfg.loader, "min_batch_size", 2)

            # Compute atom counts from systems
            print("Computing atom counts for dynamic batching...")
            sizes = self._compute_atom_counts(self.systems["train"])

            # Check if numpy is available (it should be in this env)
            import numpy as np

            # Filter outliers: keep middle 98% (drop bottom 1% and top 1%)
            p1 = np.percentile(sizes, 1)
            p99 = np.percentile(sizes, 99)
            print(
                f"Filtering training systems by size: keeping range [{p1:.1f}, {p99:.1f}] (1%-99%)"
            )

            # We enforce the percentile range
            valid_indices = [i for i, s in enumerate(sizes) if s >= p1 and s <= p99]

            if len(valid_indices) < len(sizes):
                print(
                    f"Filtering {len(sizes) - len(valid_indices)} systems (size outside [{p1:.1f}, {p99:.1f}])."
                )
                filtered_systems = [self.systems["train"][i] for i in valid_indices]
                filtered_sizes = [sizes[i] for i in valid_indices]
            else:
                filtered_systems = self.systems["train"]
                filtered_sizes = sizes

            dataset = self._create_dataset("train", apply_noise=True)
            # Hack: replace systems in the dataset with filtered ones if needed
            # A cleaner way would be to pass systems to create_dataset, but we'll re-instantiate or set attribute
            dataset.systems = filtered_systems

            batch_sampler = AtomBudgetBatchSampler(
                sizes=filtered_sizes,
                max_atoms=max_atoms,
                min_batch_size=min_batch_size,
                shuffle=True,
            )

            # Remove batch_size from loader_args for batch_sampler
            loader_args = {
                k: v for k, v in self.loader_args.items() if k not in ["batch_size"]
            }
            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_args)
        else:
            dataset = self._create_dataset("train", apply_noise=True)
            shuffle = getattr(self.cfg.loader, "shuffle", True)
            return DataLoader(dataset, shuffle=shuffle, **self.loader_args)

    def _compute_atom_counts(self, systems) -> list:
        """Compute total atom count for each system (R + L)."""
        sizes = []
        for sys in systems:
            # Try to use explicit path if available, else fallback to constructed path
            r_path = sys.get("receptor_path")
            if not r_path:
                r_path = os.path.join(self.pdb_dir, f"{sys['receptor_id']}.pdb")

            l_path = sys.get("ligand_path")
            if not l_path:
                l_path = os.path.join(self.pdb_dir, f"{sys['ligand_id']}.pdb")

            total = self._count_atoms(r_path) + self._count_atoms(l_path)
            sizes.append(total)
        return sizes

    @staticmethod
    def _count_atoms(pdb_path: str) -> int:
        """Count ATOM records in a PDB file."""
        if not os.path.exists(pdb_path):
            return 0
        count = 0
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    count += 1
        return count

    def val_dataloader(self) -> DataLoader:
        return self._eval_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader("test")

    def _eval_dataloader(self, split: str) -> DataLoader:
        """Build a DataLoader for validation or test (no shuffle, no percentile filtering)."""
        use_dynamic_batching = getattr(self.cfg.loader, "use_dynamic_batching", False)
        dataset = self._create_dataset(split, apply_noise=False)

        if use_dynamic_batching:
            max_atoms = getattr(self.cfg.loader, "max_atoms_per_batch", 50000)
            sizes = self._compute_atom_counts(self.systems[split])

            batch_sampler = AtomBudgetBatchSampler(
                sizes=sizes,
                max_atoms=max_atoms,
                min_batch_size=1,
                shuffle=False,
            )

            loader_args = {
                k: v for k, v in self.loader_args.items() if k not in ["batch_size"]
            }
            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_args)
        else:
            return DataLoader(dataset, shuffle=False, **self.loader_args)
