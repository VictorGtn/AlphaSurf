import os
import pickle
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(script_dir, "..", "..", ".."))

from alphasurf.utils.batch_sampler import AtomBudgetBatchSampler
from alphasurf.utils.data_utils import (
    AtomBatch,
    GraphLoader,
    SurfaceLoader,
    update_model_input_dim,
)

ligands = ["ADP", "COA", "FAD", "HEM", "NAD", "NAP", "SAM"]
type_idx = {type_: ix for ix, type_ in enumerate(ligands)}


def get_systems_from_ligands(
    split_list_path, ligands_path, out_path=None, recompute=False
):
    if out_path is not None:
        if os.path.exists(out_path) and not recompute:
            all_pockets = pickle.load(open(out_path, "rb"))
            return all_pockets
    all_pockets = {}
    split_list = open(split_list_path).read().splitlines()
    for pdb_chains in split_list:
        pdb = pdb_chains.split("_")[0]
        ligand_coords = np.load(
            os.path.join(ligands_path, f"{pdb}_ligand_coords.npy"),
            allow_pickle=True,
            encoding="bytes",
        )  # .astype(np.float32)
        ligand_types = np.load(os.path.join(ligands_path, f"{pdb}_ligand_types.npy"))
        ligand_types = [lig.decode() for lig in ligand_types]
        for ix, (lig_type, lig_coord) in enumerate(zip(ligand_types, ligand_coords)):
            lig_coord = lig_coord.astype(np.float32)
            pocket = f"{pdb_chains}_patch_{ix}_{lig_type}"
            all_pockets[pocket] = np.reshape(lig_coord, (-1, 3)), type_idx[lig_type]
    if out_path is not None:
        pickle.dump(all_pockets, open(out_path, "wb"))
    return all_pockets


class SurfaceLoaderMasifLigand(SurfaceLoader):
    """
    Surface loader for MaSIF-Ligand that supports augmented views.

    When n_augmented_views > 1, this loader will randomly select one of the
    pre-computed augmented views at each load call. This provides data augmentation
    without increasing the number of samples per epoch.

    Args:
        config: Configuration object with the following relevant fields:
            - use_whole_surfaces: If True, load full protein surface instead of patch
            - n_augmented_views: Number of pre-computed augmented views (default: 1)
            - augmentation_sigma: Sigma used during preprocessing (for directory naming)
            - augmentation_noise_type: Type of noise ('normal' or 'isotropic')
    """

    def __init__(self, config):
        # Store augmentation params before calling super().__init__
        self.n_augmented_views = getattr(config, "n_augmented_views", 1)
        self.augmentation_sigma = getattr(config, "augmentation_sigma", 0.3)
        self.augmentation_noise_type = getattr(
            config, "augmentation_noise_type", "normal"
        )

        # If using augmentation, modify data_name to point to augmented directory
        if self.n_augmented_views > 1:
            # Transform data_name to include augmentation suffix
            # e.g., "surf_1.0_False" -> "surf_1.0_False_aug20_normal_sigma0.3"
            original_data_name = config.data_name
            config = self._clone_config_with_aug_dir(config)

        super().__init__(config)

    def _clone_config_with_aug_dir(self, config):
        """Create a modified config with augmentation directory name."""
        # Create a mutable copy of the config
        from torch_geometric.data import Data

        new_config = Data()
        for key in dir(config):
            if not key.startswith("_"):
                try:
                    setattr(new_config, key, getattr(config, key))
                except:
                    pass

        # Modify data_name to point to augmented directory
        # Format: {original}_aug{n_views}_{noise_type}_sigma{sigma}
        original_name = config.data_name
        new_config.data_name = f"{original_name}_aug{self.n_augmented_views}_{self.augmentation_noise_type}_sigma{self.augmentation_sigma}"
        return new_config

    def load(self, pocket_name):
        if self.config.use_whole_surfaces:
            pocket_name = pocket_name.split("_patch_")[0]

        # If using augmented views, randomly select one
        if self.n_augmented_views > 1:
            view_idx = np.random.randint(0, self.n_augmented_views)
            pocket_name = f"{pocket_name}_view_{view_idx}"

        return super().load(pocket_name)


class GraphLoaderMasifLigand(GraphLoader):
    def __init__(self, config):
        super().__init__(config)

    def load(self, pocket_name):
        pocket_name = pocket_name.split("_patch_")[0]
        return super().load(pocket_name)


class MasifLigandDataset(Dataset):
    def __init__(self, systems, surface_builder, graph_builder):
        self.systems = systems
        self.systems_keys = list(systems.keys())
        self.surface_builder = surface_builder
        self.graph_builder = graph_builder

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        pocket = self.systems_keys[idx]
        # pocket = "1DW1_A_patch_0_HEM"
        lig_coord, lig_type = self.systems[pocket]
        lig_coord = torch.from_numpy(lig_coord)
        # pocket = f'{pdb_chains}_patch_{ix}_{lig_type}'
        surface = self.surface_builder.load(pocket)
        graph = self.graph_builder.load(pocket)
        if surface is None or graph is None:
            return None
        if (
            hasattr(surface, "x")
            and surface.x is not None
            and torch.isnan(surface.x).any()
        ) or (
            hasattr(graph, "x") and graph.x is not None and torch.isnan(graph.x).any()
        ):
            return None
        item = Data(surface=surface, graph=graph, lig_coord=lig_coord, label=lig_type)
        return item


class MasifLigandDataset_InMemory(Dataset):
    def __init__(self, dataset_dir):
        self.systems = torch.load(dataset_dir, weights_only=False)

    def __len__(self):
        return len(self.systems)

    def __getitem__(self, idx):
        item = self.systems[idx]
        return item


class MasifLigandDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.surface_loader = SurfaceLoaderMasifLigand(cfg.cfg_surface)
        self.graph_loader = GraphLoaderMasifLigand(cfg.cfg_graph)
        script_dir = os.path.dirname(os.path.realpath(__file__))
        if cfg.data_dir is None:
            masif_ligand_data_dir = os.path.join(
                script_dir, "..", "..", "..", "data", "masif_ligand"
            )
        else:
            masif_ligand_data_dir = cfg.data_dir
        splits_dir = os.path.join(
            masif_ligand_data_dir, "raw_data_MasifLigand", "splits"
        )
        ligands_path = os.path.join(
            masif_ligand_data_dir, "raw_data_MasifLigand", "ligand"
        )
        self.use_inmem = cfg.use_inmem
        if self.use_inmem:
            self.train_dir = os.path.join(
                cfg.data_dir, "Inmemory_train_surfhmr_rggraph_esm.pt"
            )
            self.val_dir = os.path.join(
                cfg.data_dir, "Inmemory_val_surfhmr_rggraph_esm.pt"
            )
            self.test_dir = os.path.join(
                cfg.data_dir, "Inmemory_test_surfhmr_rggraph_esm.pt"
            )
            if not (
                os.path.isfile(self.train_dir)
                and os.path.isfile(self.val_dir)
                and os.path.isfile(self.test_dir)
            ):
                self.create_inmem_set(cfg)
        self.systems = []
        for split in ["train", "val", "test"]:
            splits_path = os.path.join(splits_dir, f"{split}-list.txt")
            out_path = os.path.join(splits_dir, f"{split}.p")
            self.systems.append(
                get_systems_from_ligands(
                    splits_path, ligands_path=ligands_path, out_path=out_path
                )
            )
        self.pdb_dir = os.path.join(
            masif_ligand_data_dir, "raw_data_MasifLigand", "pdb"
        )
        self.cfg = cfg
        prefetch_factor = getattr(self.cfg.loader, "prefetch_factor", None)
        persistent_workers = getattr(self.cfg.loader, "persistent_workers", False)
        self.loader_args = {
            "num_workers": self.cfg.loader.num_workers,
            "batch_size": self.cfg.loader.batch_size,
            "pin_memory": self.cfg.loader.pin_memory,
            "drop_last": self.cfg.loader.drop_last,
            "collate_fn": self._collate_fn,
        }
        if self.cfg.loader.num_workers and self.cfg.loader.num_workers > 0:
            if prefetch_factor is not None:
                self.loader_args["prefetch_factor"] = prefetch_factor
            self.loader_args["persistent_workers"] = persistent_workers

        if self.use_inmem:
            dataset_temp = MasifLigandDataset_InMemory(self.val_dir)
        else:
            dataset_temp = MasifLigandDataset(
                self.systems[0], self.surface_loader, self.graph_loader
            )
        update_model_input_dim(cfg, dataset_temp=dataset_temp)

    @staticmethod
    def _collate_fn(x):
        return AtomBatch.from_data_list(x)

    def create_inmem_set(self, cfg):
        train_set = MasifLigandDataset(
            self.systems[0], self.surface_loader, self.graph_loader
        )
        val_set = MasifLigandDataset(
            self.systems[1], self.surface_loader, self.graph_loader
        )
        test_set = MasifLigandDataset(
            self.systems[2], self.surface_loader, self.graph_loader
        )
        train_list = []
        val_list = []
        test_list = []
        for data in train_set:
            train_list.append(data)
        torch.save(
            train_list,
            os.path.join(cfg.data_dir, "Inmemory_train_surfhmr_rggraph_esm.pt"),
        )
        for data in val_set:
            val_list.append(data)
        torch.save(
            val_list, os.path.join(cfg.data_dir, "Inmemory_val_surfhmr_rggraph_esm.pt")
        )
        for data in test_set:
            test_list.append(data)
        torch.save(
            test_list,
            os.path.join(cfg.data_dir, "Inmemory_test_surfhmr_rggraph_esm.pt"),
        )

    def train_dataloader(self):
        if self.use_inmem:
            dataset = MasifLigandDataset_InMemory(self.train_dir)
        else:
            dataset = MasifLigandDataset(
                self.systems[0], self.surface_loader, self.graph_loader
            )

        # Check for dynamic batching config
        use_dynamic_batching = getattr(self.cfg.loader, "use_dynamic_batching", False)

        if use_dynamic_batching and not self.use_inmem:
            max_atoms = getattr(self.cfg.loader, "max_atoms_per_batch", 50000)
            min_batch_size = getattr(self.cfg.loader, "min_batch_size", 2)

            # Compute sizes
            print("Computing atom counts for dynamic batching...")
            sizes = self._compute_atom_counts(dataset.systems_keys)

            # Create sampler
            batch_sampler = AtomBudgetBatchSampler(
                sizes=sizes,
                max_atoms=max_atoms,
                min_batch_size=min_batch_size,
                shuffle=self.cfg.loader.shuffle,
            )

            # Prepare loader args (remove batch_size, shuffle, drop_last)
            loader_args = self.loader_args.copy()
            loader_args.pop("batch_size", None)
            loader_args.pop("drop_last", None)

            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_args)

        return DataLoader(dataset, shuffle=self.cfg.loader.shuffle, **self.loader_args)

    def _compute_atom_counts(self, pocket_names):
        """Compute atom counts for the PDBs associated with each pocket."""
        sizes = []
        pdb_cache = {}

        for pocket in pocket_names:
            # Pocket format: pdb_chain_patch_idx_ligandType
            # Example: 1DW1_A_patch_0_HEM
            # We assume the PDB file is named {pdb_chain}.pdb
            # Or maybe just {pdb}.pdb? Based on list_dir, many are like 1A27_AB.pdb
            # The split list contains entries like "1DW1_A", which likely map to filenames.

            # Extract the PDB identifier from the pocket name
            # The split list variable in get_systems_from_ligands is `pdb_chains`
            # `pocket = f"{pdb_chains}_patch_{ix}_{lig_type}"`
            # So splitting by `_patch_` and taking the first part gives `pdb_chains`
            pdb_chains = pocket.split("_patch_")[0]

            if pdb_chains in pdb_cache:
                sizes.append(pdb_cache[pdb_chains])
                continue

            pdb_path = os.path.join(self.pdb_dir, f"{pdb_chains}.pdb")
            count = 0
            if os.path.exists(pdb_path):
                with open(pdb_path, "r") as f:
                    for line in f:
                        if line.startswith("ATOM"):
                            count += 1
            else:
                # If exact match fails, try splitting by _ again?
                # But looking at listing, they match the pattern (e.g. 1A27_AB.pdb)
                pass

            pdb_cache[pdb_chains] = count
            sizes.append(count)

        return sizes

    def val_dataloader(self):
        if self.use_inmem:
            dataset = MasifLigandDataset_InMemory(self.val_dir)
        else:
            dataset = MasifLigandDataset(
                self.systems[1], self.surface_loader, self.graph_loader
            )

        use_dynamic_batching = getattr(self.cfg.loader, "use_dynamic_batching", False)

        if use_dynamic_batching and not self.use_inmem:
            max_atoms = getattr(self.cfg.loader, "max_atoms_per_batch", 50000)
            min_batch_size = 1  # Allow batch size of 1 for validation

            # Compute sizes
            sizes = self._compute_atom_counts(dataset.systems_keys)

            # Create sampler (no shuffle for validation)
            batch_sampler = AtomBudgetBatchSampler(
                sizes=sizes,
                max_atoms=max_atoms,
                min_batch_size=min_batch_size,
                shuffle=False,
            )

            # Prepare loader args
            loader_args = self.loader_args.copy()
            loader_args.pop("batch_size", None)
            loader_args.pop("drop_last", None)

            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_args)

        return DataLoader(dataset, shuffle=False, **self.loader_args)

    def test_dataloader(self):
        if self.use_inmem:
            dataset = MasifLigandDataset_InMemory(self.test_dir)
        else:
            dataset = MasifLigandDataset(
                self.systems[2], self.surface_loader, self.graph_loader
            )

        use_dynamic_batching = getattr(self.cfg.loader, "use_dynamic_batching", False)

        if use_dynamic_batching and not self.use_inmem:
            max_atoms = getattr(self.cfg.loader, "max_atoms_per_batch", 50000)
            min_batch_size = 1  # Allow batch size of 1 for test

            # Compute sizes
            sizes = self._compute_atom_counts(dataset.systems_keys)

            # Create sampler (no shuffle for test)
            batch_sampler = AtomBudgetBatchSampler(
                sizes=sizes,
                max_atoms=max_atoms,
                min_batch_size=min_batch_size,
                shuffle=False,
            )

            # Prepare loader args
            loader_args = self.loader_args.copy()
            loader_args.pop("batch_size", None)
            loader_args.pop("drop_last", None)

            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_args)

        return DataLoader(dataset, shuffle=False, **self.loader_args)


if __name__ == "__main__":
    pass
    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(
        script_dir, "..", "..", "..", "data", "masif_ligand"
    )
    splits_dir = os.path.join(masif_ligand_data_dir, "raw_data_MasifLigand", "splits")
    ligands_path = os.path.join(masif_ligand_data_dir, "raw_data_MasifLigand", "ligand")
    # for split in ['train', 'val', 'test']:
    #     splits_path = os.path.join(splits_dir, f'{split}-list.txt')
    #     out_path = os.path.join(splits_dir, f'{split}.p')
    #     systems = get_systems_from_ligands(splits_path,
    #                                        ligands_path=ligands_path,
    #                                        out_path=out_path,
    #                                        recompute=True)

    # SURFACE
    script_dir = os.path.dirname(os.path.realpath(__file__))
    masif_ligand_data_dir = os.path.join(
        script_dir, "..", "..", "..", "data", "masif_ligand"
    )
    cfg_surface = Data()
    cfg_surface.use_surfaces = True
    cfg_surface.feat_keys = "all"
    cfg_surface.oh_keys = "all"
    cfg_surface.use_whole_surfaces = False
    # cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_hmr')
    cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, "surf_ours")
    # cfg_surface.use_whole_surfaces = True
    # cfg_surface.data_dir = os.path.join(masif_ligand_data_dir, 'surf_full')
    surface_loader = SurfaceLoaderMasifLigand(cfg_surface)

    # GRAPHS
    cfg_graph = Data()
    cfg_graph.use_graphs = False
    cfg_graph.feat_keys = "all"
    cfg_graph.oh_keys = "all"
    cfg_graph.esm_dir = "toto"
    cfg_graph.use_esm = False
    cfg_graph.data_dir = os.path.join(masif_ligand_data_dir, "rgraph")
    # cfg_graph.data_dir= os.path.join(masif_ligand_data_dir, 'agraph')
    graph_loader = GraphLoaderMasifLigand(cfg_graph)

    split = "train"
    splits_path = os.path.join(splits_dir, f"{split}-list.txt")
    out_path = os.path.join(splits_dir, f"{split}.p")
    systems = get_systems_from_ligands(
        splits_path, ligands_path=ligands_path, out_path=out_path
    )
    dataset = MasifLigandDataset(systems, surface_loader, graph_loader)
    a = dataset[0]

    loader_cfg = Data(
        num_workers=0, batch_size=4, pin_memory=False, prefetch_factor=2, shuffle=False
    )
    simili_cfg = Data(cfg_surface=cfg_surface, cfg_graph=cfg_graph, loader=loader_cfg)
    datamodule = MasifLigandDataModule(cfg=simili_cfg)
    loader = datamodule.train_dataloader()
    for i, batch in enumerate(loader):
        if i > 3:
            break
