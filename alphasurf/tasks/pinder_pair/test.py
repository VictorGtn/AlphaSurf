"""
Testing script for PINDER-Pair protein-protein interaction task.
Supports alignment-based testing for Apo and AF2 structures.
"""

import os
import sys
import warnings
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).absolute().parents[3]))

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from alphasurf.protein.protein_loader import ProteinLoader
from alphasurf.tasks.pinder_pair.dataset import (
    PinderAlignedDataset,
    PinderPairDataset,
    load_pinder_split,
)
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule
from alphasurf.utils.data_utils import AtomBatch, update_model_input_dim
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    # Enable evaluation resolver
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    pl.seed_everything(cfg.seed, workers=True)

    test_setting = getattr(cfg, "test_setting", "apo")
    print(
        f"\n{'=' * 60}\nRUNNING PINDER-PAIR TEST (Setting: {test_setting})\n{'=' * 60}\n"
    )

    # 1. Load Systems
    data_dir = cfg.data_dir
    eval_split = os.environ.get("EVAL_SPLIT", "test")
    print(f"Loading {eval_split} split for setting: {test_setting}")

    target_systems = load_pinder_split(
        data_dir,
        split=eval_split,
        test_setting=test_setting if eval_split == "test" else None,
        max_systems=None,
    )
    print(f"Index loaded: {len(target_systems)} systems")

    # 2. Setup Dataset Class
    DatasetClass = PinderPairDataset
    if test_setting in ["apo", "af2"]:
        print("Using PinderAlignedDataset for alignment based on Holo ground truth.")
        DatasetClass = PinderAlignedDataset

        # Load Holo systems to get reference paths
        print("Loading Holo reference paths...")
        holo_systems = load_pinder_split(
            data_dir, split="test", test_setting="holo", max_systems=None
        )
        holo_map = {s["id"]: s for s in holo_systems}

        # Merge Holo paths into target systems
        merged_count = 0
        for s in target_systems:
            if s["id"] in holo_map:
                s["holo_receptor_path"] = holo_map[s["id"]].get("receptor_path")
                s["holo_ligand_path"] = holo_map[s["id"]].get("ligand_path")
                s["holo_receptor_id"] = holo_map[s["id"]].get("receptor_id")
                s["holo_ligand_id"] = holo_map[s["id"]].get("ligand_id")
                merged_count += 1
            else:
                pass
                # print(f"Warning: Holo reference not found for {s['id']}")

        print(f"Merged Holo paths for {merged_count}/{len(target_systems)} systems.")

    # 3. Initialize ProteinLoader MANUALLY (bypass DataModule)
    print("Initializing ProteinLoader...")

    from alphasurf.utils.config_utils import merge_surface_config

    on_fly_cfg = getattr(cfg, "on_fly", None)
    mode = "on_fly" if on_fly_cfg is not None else "disk"
    pdb_dir = os.path.join(data_dir, "pdb")

    surface_dir = None
    graph_dir = None
    if mode == "disk":
        surface_dir = os.path.join(cfg.cfg_surface.data_dir, cfg.cfg_surface.data_name)
        graph_dir = os.path.join(cfg.cfg_graph.data_dir, cfg.cfg_graph.data_name)
    esm_dir = getattr(cfg.cfg_graph, "esm_dir", None)

    noise_augmentor = None  # No noise for test

    surface_config = merge_surface_config(cfg.cfg_surface, on_fly_cfg)
    graph_config = merge_surface_config(cfg.cfg_graph, on_fly_cfg)

    protein_loader = ProteinLoader(
        mode=mode,
        pdb_dir=pdb_dir,
        surface_dir=surface_dir,
        graph_dir=graph_dir,
        esm_dir=esm_dir,
        surface_config=surface_config,
        graph_config=graph_config,
        noise_augmentor=noise_augmentor,
    )

    # Create Dataset
    print("=== DATASET PARAMS ===")
    print(f"interface_distance_graph: {cfg.interface_distance_graph}")
    print(f"interface_distance_surface: {cfg.interface_distance_surface}")
    print(f"neg_to_pos_ratio: {cfg.neg_to_pos_ratio}")
    print(f"surface_neg_to_pos_ratio: {cfg.surface_neg_to_pos_ratio}")
    print()
    test_dataset = DatasetClass(
        systems=target_systems,
        protein_loader=protein_loader,
        pdb_dir=pdb_dir,
        apply_noise=False,
        neg_to_pos_ratio=float(cfg.neg_to_pos_ratio),
        max_pos_per_pair=int(cfg.max_pos_per_pair),
        interface_distance_graph=float(cfg.interface_distance_graph),
        interface_distance_surface=float(cfg.interface_distance_surface),
        surface_neg_to_pos_ratio=float(cfg.surface_neg_to_pos_ratio),
    )

    # Calculate Input Dims if no checkpoint
    # We use the FIRST VALID SAMPLE from test dataset to infer dims
    # Since we are verifying, we MUST ensure dims are set.
    print("Inferring model dimensions from first valid sample...")

    # We might need to try a few indices if first one is None
    temp_dataset_subset = torch.utils.data.Subset(
        test_dataset, range(min(len(test_dataset), 20))
    )
    try:
        update_model_input_dim(
            cfg, dataset_temp=temp_dataset_subset, gkey="graph_1", skey="surface_1"
        )
    except Exception as e:
        print(f"Warning: Could not infer input dims from test subset: {e}")
        # If we have checkpoint, we might be fine. If not, model init will fail/be wrong.

    # Collate fn
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return AtomBatch.from_data_list(batch)

    # Create DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.loader.pin_memory,
    )

    # 4. Load Model
    print("Loading model...")
    ckpt_path = getattr(cfg, "ckpt_path", None)
    if not ckpt_path:
        print(
            "Warning: No checkpoint path specified (cfg.ckpt_path). Using random weights."
        )
        model = PinderPairModule(cfg)
    else:
        print(f"Loading checkpoint: {ckpt_path}")
        model = PinderPairModule.load_from_checkpoint(ckpt_path, cfg=cfg)

    # Diagnostic: verify checkpoint loaded real weights (not random)
    param_stats = {}
    for name, param in model.named_parameters():
        param_stats[name] = (param.mean().item(), param.std().item(), param.shape)
    print(f"\nModel has {len(param_stats)} parameters")
    for name, (mean, std, shape) in list(param_stats.items())[:5]:
        print(f"  {name}: shape={shape}, mean={mean:.6f}, std={std:.6f}")
    print()

    # 5. Device setup
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.device}")
    model = model.to(device)

    model.eval()

    # Print training config from checkpoint for comparison
    if ckpt_path:
        ckpt_data = torch.load(ckpt_path, map_location="cpu")
        if "hyper_parameters" in ckpt_data:
            hp = ckpt_data["hyper_parameters"].get("cfg", {})
            print("=== TRAINING CONFIG ===")
            for key in [
                "neg_to_pos_ratio",
                "interface_distance_graph",
                "interface_distance_surface",
                "surface_neg_to_pos_ratio",
            ]:
                print(f"{key}: {hp.get(key, 'N/A')}")
            print()
        del ckpt_data

    # 6. Manual test loop with per-system and global AUROC
    print(f"Starting testing (eval_split={eval_split}, model.eval())...")
    all_logits = []
    all_labels = []
    system_aurocs = []
    n_batches = 0
    n_skipped = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                n_skipped += 1
                continue

            batch = batch.to(device)
            if batch.num_graphs < getattr(cfg, "min_batch_size", 2):
                n_skipped += 1
                continue

            # Get labels (keep per-system structure for per-system AUROC)
            if isinstance(batch.label, list):
                per_system_labels = [l.reshape(-1) for l in batch.label]
                labels = torch.cat(batch.label).reshape(-1, 1)
            else:
                per_system_labels = None
                labels = batch.label.reshape(-1, 1)

            outputs, outputs_surf = model(batch)

            all_logits.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

            # Per-system AUROC
            if per_system_labels is not None:
                offset = 0
                sys_ids = (
                    batch.system_id
                    if hasattr(batch, "system_id")
                    else [None] * len(per_system_labels)
                )
                if isinstance(sys_ids, str):
                    sys_ids = [sys_ids]
                for si, sys_labels in enumerate(per_system_labels):
                    n_pairs = len(sys_labels)
                    sys_logits = (
                        outputs[offset : offset + n_pairs]
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()
                    )
                    sys_lab = sys_labels.cpu().numpy().astype(int).ravel()
                    n_pos = sys_lab.sum()
                    n_neg = len(sys_lab) - n_pos
                    if n_pos > 0 and n_neg > 0:
                        sys_auroc = roc_auc_score(sys_lab, sys_logits)
                        sid = sys_ids[si] if si < len(sys_ids) else "?"
                        system_aurocs.append((sid, sys_auroc, n_pos, n_neg))
                    offset += n_pairs

            # Print diagnostics for first few batches
            if batch_idx < 3:
                print(
                    f"\n  Batch {batch_idx}: {batch.num_graphs} graphs, "
                    f"{len(labels)} pairs"
                )
                print(
                    f"    Labels: {labels.sum().item():.0f} pos / "
                    f"{(1 - labels).sum().item():.0f} neg"
                )
                print(
                    f"    Logits: min={outputs.min().item():.4f}, "
                    f"max={outputs.max().item():.4f}, "
                    f"mean={outputs.mean().item():.4f}, "
                    f"std={outputs.std().item():.4f}"
                )

            n_batches += 1
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"  Processed {batch_idx}/{len(test_loader)} batches...")

    print(f"\nProcessed {n_batches} batches, skipped {n_skipped}")

    if all_logits:
        all_logits = torch.cat(all_logits).numpy().ravel()
        all_labels = torch.cat(all_labels).numpy().astype(int).ravel()

        print(f"Total predictions: {len(all_logits)}")
        print(
            f"Total positives: {all_labels.sum()}, negatives: {(1 - all_labels).sum()}"
        )
        print(
            f"Logit stats: min={all_logits.min():.4f}, "
            f"max={all_logits.max():.4f}, "
            f"mean={all_logits.mean():.4f}, "
            f"std={all_logits.std():.4f}"
        )

        global_auroc = roc_auc_score(all_labels, all_logits)
        print(f"\n>>> GLOBAL AUROC: {global_auroc:.6f} <<<")

        # Per-system AUROC distribution
        if system_aurocs:
            aurocs = [a for _, a, _, _ in system_aurocs]
            aurocs_arr = np.array(aurocs)
            print(f"\n=== PER-SYSTEM AUROC ({len(aurocs)} systems) ===")
            print(
                f"Mean: {aurocs_arr.mean():.4f}, Median: {np.median(aurocs_arr):.4f}, "
                f"Std: {aurocs_arr.std():.4f}"
            )
            print(f"Min: {aurocs_arr.min():.4f}, Max: {aurocs_arr.max():.4f}")
            # Distribution buckets
            for threshold in [0.6, 0.7, 0.8, 0.9]:
                count = (aurocs_arr >= threshold).sum()
                print(
                    f"  Systems with AUROC >= {threshold}: {count}/{len(aurocs)} "
                    f"({100 * count / len(aurocs):.1f}%)"
                )
            # Top 10 and bottom 10 systems
            sorted_sys = sorted(system_aurocs, key=lambda x: x[1], reverse=True)
            print("\nTop 10 systems:")
            for sid, auc, np_, nn_ in sorted_sys[:10]:
                print(f"  {sid}: AUROC={auc:.4f} ({np_} pos, {nn_} neg)")
            print("\nBottom 10 systems:")
            for sid, auc, np_, nn_ in sorted_sys[-10:]:
                print(f"  {sid}: AUROC={auc:.4f} ({np_} pos, {nn_} neg)")
    else:
        print("No valid batches processed!")


if __name__ == "__main__":
    main()
