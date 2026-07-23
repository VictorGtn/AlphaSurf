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
from torch.utils.data import DataLoader, Dataset


class _RetryDataset(Dataset):
    """Wrapper that retries next indices on failure (TIMING mode only)."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        for offset in range(len(self.dataset)):
            probe = (idx + offset) % len(self.dataset)
            result = self.dataset[probe]
            if result is not None:
                return result
        return None


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

        # Filter test split to common systems (shared across all surface methods)
        common_only = getattr(cfg, "common_only", False)
        if common_only and self.test_setting:
            common_csv = os.path.join(
                data_dir, f"common_systems_{self.test_setting}.csv"
            )
            if os.path.exists(common_csv):
                import pandas as pd

                common_df = pd.read_csv(common_csv)
                common_ids = set(common_df["id"])
                n_before = len(self.systems["test"])
                self.systems["test"] = [
                    s for s in self.systems["test"] if s["id"] in common_ids
                ]
                print(
                    f"Common-only filter: {len(self.systems['test'])}/{n_before} systems "
                    f"({self.test_setting})"
                )
            else:
                print(
                    f"WARNING: common_only=True but {common_csv} not found. "
                    f"Using full test set."
                )

        # Merge holo reference paths into test systems for apo/af2 alignment
        if self.test_setting in ["apo", "af2"]:
            holo_systems = load_pinder_split(
                data_dir, split="test", test_setting="holo"
            )
            holo_map = {s["id"]: s for s in holo_systems}
            merged = 0
            for s in self.systems["test"]:
                if s["id"] in holo_map:
                    s["holo_receptor_path"] = holo_map[s["id"]].get("receptor_path")
                    s["holo_ligand_path"] = holo_map[s["id"]].get("ligand_path")
                    s["holo_receptor_id"] = holo_map[s["id"]].get("receptor_id")
                    s["holo_ligand_id"] = holo_map[s["id"]].get("ligand_id")
                    merged += 1
            print(
                f"Merged Holo paths for {merged}/{len(self.systems['test'])} test systems."
            )

        # Build protein loaders: train gets the noise augmentor, eval does not.
        # Datasets pick cache branches by inspecting loader.noise_augmentor.
        (
            self.protein_loader_train,
            self.protein_loader_eval,
        ) = self._build_protein_loaders(cfg)

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

        # Precomputed interface directory (disk mode)
        self.interface_dir = self._resolve_interface_dir(cfg)

        # DataLoader args
        self.loader_args = self._build_loader_args(cfg)

        # Resolve surface label mode from config
        if getattr(cfg, "on_fly", None) is not None:
            surface_label_mode = getattr(cfg.on_fly, "surface_label_mode", "atom")
        else:
            surface_label_mode = getattr(cfg, "surface_label_mode", "atom")

        # Update model input dimensions
        temp_dataset = PinderPairDataset(
            systems=self.systems["train"][:10],
            protein_loader=self.protein_loader_eval,
            pdb_dir=self.pdb_dir,
            interface_distance_graph=self.interface_distance_graph,
            interface_distance_surface=self.interface_distance_surface,
        )
        temp_dataset.SURFACE_LABEL_MODE = surface_label_mode
        update_model_input_dim(
            cfg, dataset_temp=temp_dataset, gkey="graph_1", skey="surface_1"
        )

    @staticmethod
    def _resolve_interface_dir(cfg) -> Optional[str]:
        """Resolve precomputed interface directory from config."""
        cfg_interface = getattr(cfg, "cfg_interface", None)
        if cfg_interface is None:
            return None
        data_dir = getattr(cfg_interface, "data_dir", cfg.data_dir)
        data_name = getattr(cfg_interface, "data_name", None)
        if data_name is None:
            return None
        return os.path.join(data_dir, data_name)

    def _build_protein_loaders(self, cfg) -> tuple:
        """Build train (with noise) and eval (no noise) ProteinLoaders."""
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

        noise_augmentor = self._build_noise_augmentor(cfg)

        # Merge configs
        surface_config = self._merge_config(cfg.cfg_surface, on_fly_cfg)
        graph_config = self._merge_config(cfg.cfg_graph, on_fly_cfg)

        surface_config.use_poisson = "poisson" in cfg.encoder.name

        common_kwargs = dict(
            mode=mode,
            pdb_dir=self.pdb_dir,
            surface_dir=surface_dir,
            graph_dir=graph_dir,
            esm_dir=esm_dir,
            surface_config=surface_config,
            graph_config=graph_config,
        )
        loader_train = ProteinLoader(noise_augmentor=noise_augmentor, **common_kwargs)
        loader_eval = ProteinLoader(noise_augmentor=None, **common_kwargs)
        return loader_train, loader_eval

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

    @staticmethod
    def _maybe_wrap_timing(dataset):
        if os.environ.get("TIMING", "0") == "1":
            return _RetryDataset(dataset)
        return dataset

    def _create_dataset(self, split: str) -> PinderPairDataset:
        DatasetClass = PinderPairDataset
        # Use aligned dataset for Apo/AF2 testing to transfer labels from Holo
        if split == "test" and self.test_setting in ["apo", "af2"]:
            DatasetClass = PinderAlignedDataset

        protein_loader = (
            self.protein_loader_train if split == "train" else self.protein_loader_eval
        )

        if self.cfg.on_fly is not None:
            surface_label_mode = getattr(self.cfg.on_fly, "surface_label_mode", "atom")
        else:
            surface_label_mode = getattr(self.cfg, "surface_label_mode", "atom")

        # Multi-member cluster sampling only applies to train; val/test are
        # singletons by PINDER design and PinderAlignedDataset relies on
        # one-to-one alignment.
        cluster_sampling = (
            getattr(self.cfg, "cluster_sampling", "single")
            if split == "train"
            else "single"
        )

        dataset = DatasetClass(
            systems=self.systems[split],
            protein_loader=protein_loader,
            pdb_dir=self.pdb_dir,
            neg_to_pos_ratio=self.neg_to_pos_ratio,
            max_pos_per_pair=self.max_pos_per_pair,
            interface_distance_graph=self.interface_distance_graph,
            interface_distance_surface=self.interface_distance_surface,
            surface_neg_to_pos_ratio=self.surface_neg_to_pos_ratio,
            interface_dir=self.interface_dir,
            cluster_sampling=cluster_sampling,
        )
        dataset.SURFACE_LABEL_MODE = surface_label_mode
        return dataset

    def train_dataloader(self) -> DataLoader:
        # Use dynamic batch sampler if configured
        use_dynamic_batching = getattr(self.cfg.loader, "use_dynamic_batching", False)

        if use_dynamic_batching:
            max_atoms = getattr(self.cfg.loader, "max_atoms_per_batch", 50000)
            min_batch_size = getattr(self.cfg.loader, "min_batch_size", 2)

            dataset = self._create_dataset("train")

            # Compute atom counts from systems
            print("Computing atom counts for dynamic batching...")
            sizes = self._compute_atom_counts(self.systems["train"], dataset)

            import numpy as np

            p1 = np.percentile(sizes, 1)
            p99 = np.percentile(sizes, 99)
            print(
                f"Filtering training systems by size: keeping range [{p1:.1f}, {p99:.1f}] (1%-99%)"
            )

            keep_mask = [p1 <= s <= p99 for s in sizes]
            n_dropped = len(sizes) - sum(keep_mask)
            if n_dropped:
                print(
                    f"Filtering {n_dropped} samples (size outside [{p1:.1f}, {p99:.1f}])."
                )

            if dataset._clusters is not None:
                retained_clusters = [
                    c for i, c in enumerate(dataset._clusters) if keep_mask[i]
                ]
                filtered_systems = [s for c in retained_clusters for s in c]
                filtered_sizes = [s for i, s in enumerate(sizes) if keep_mask[i]]
            else:
                filtered_systems = [
                    s for i, s in enumerate(self.systems["train"]) if keep_mask[i]
                ]
                filtered_sizes = [s for i, s in enumerate(sizes) if keep_mask[i]]

            dataset.systems = filtered_systems
            if dataset._clusters is not None:
                dataset._build_cluster_index()
            dataset.prepopulate_interface_cache()
            dataset = self._maybe_wrap_timing(dataset)

            batch_sampler = AtomBudgetBatchSampler(
                sizes=filtered_sizes,
                max_atoms=max_atoms,
                min_batch_size=min_batch_size,
                shuffle=True,
            )

            loader_args = {
                k: v for k, v in self.loader_args.items() if k not in ["batch_size"]
            }
            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_args)
        else:
            dataset = self._create_dataset("train")
            dataset.prepopulate_interface_cache()
            dataset = self._maybe_wrap_timing(dataset)
            shuffle = getattr(self.cfg.loader, "shuffle", True)
            return DataLoader(dataset, shuffle=shuffle, **self.loader_args)

    def _compute_atom_counts(self, systems, dataset) -> list:
        """Compute total atom count per dataset item.

        In single mode (or any time dataset._clusters is None), returns one
        size per system, indexed by position in `systems`.

        In multi mode, returns one size per cluster (aligned with
        dataset._clusters order), taking the max over cluster members so the
        sampler is pessimistic about which member gets drawn.

        Prefers the n_atoms_R / n_atoms_L fields written by preprocess.py;
        falls back to counting ATOM records in the PDB files.
        """
        if dataset._clusters is not None:
            return [
                max(self._system_atom_count(s, dataset) for s in cluster)
                for cluster in dataset._clusters
            ]
        return [self._system_atom_count(s, dataset) for s in systems]

    def _system_atom_count(self, sys, dataset) -> int:
        if "n_atoms_R" in sys and "n_atoms_L" in sys:
            return int(sys["n_atoms_R"]) + int(sys["n_atoms_L"])
        r_path = dataset._get_pdb_path(sys, "receptor")
        l_path = dataset._get_pdb_path(sys, "ligand")
        return self._count_atoms(r_path) + self._count_atoms(l_path)

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

    def val_dataloader(self):
        return [self._eval_dataloader("val"), self._eval_dataloader("test")]

    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader("test")

    def _eval_dataloader(self, split: str) -> DataLoader:
        """Build a DataLoader for validation or test (no shuffle, no percentile filtering)."""
        use_dynamic_batching = getattr(self.cfg.loader, "use_dynamic_batching", False)
        dataset = self._create_dataset(split)
        dataset = self._maybe_wrap_timing(dataset)

        if use_dynamic_batching:
            max_atoms = getattr(self.cfg.loader, "max_atoms_per_batch", 50000)
            sizes = self._compute_atom_counts(self.systems[split], dataset)

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
