"""
Testing script for PINDER-Pair protein-protein interaction task.
Supports alignment-based testing for Apo and AF2 structures.
"""

import os
import sys
import warnings
from pathlib import Path

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
from alphasurf.utils.data_utils import AtomBatch
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")


def _is_homodimer(system_id):
    """Determine if a system is a homodimer from its ID (e.g. '8iyi__A1_Q6CVU4--8iyi__B1_Q6CVU4')."""
    try:
        parts = system_id.split("--")
        if len(parts) != 2:
            return False
        r_uniprot = parts[0].split("_")[-1]
        l_uniprot = parts[1].split("_")[-1]
        return r_uniprot == l_uniprot
    except Exception:
        return False


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return AtomBatch.from_data_list(batch)


def setup_test_loader(cfg, test_setting):
    """Create test DataLoader for a given setting (holo/apo/af2)."""
    data_dir = cfg.data_dir
    pdb_dir = os.path.join(data_dir, "pdb")

    target_systems = load_pinder_split(
        data_dir, split="test", test_setting=test_setting
    )

    common_only = getattr(cfg, "common_only", False)
    if common_only:
        common_csv = os.path.join(data_dir, f"common_systems_{test_setting}.csv")
        if os.path.exists(common_csv):
            import pandas as pd

            common_ids = set(pd.read_csv(common_csv)["id"])
            n_before = len(target_systems)
            target_systems = [s for s in target_systems if s["id"] in common_ids]
            print(
                f"  Common-only filter: {len(target_systems)}/{n_before} systems "
                f"({test_setting})"
            )
        else:
            print(
                f"  WARNING: common_only=True but {common_csv} not found. "
                f"Using full test set."
            )

    print(f"  {test_setting}: {len(target_systems)} systems")

    DatasetClass = PinderPairDataset
    if test_setting in ["apo", "af2"]:
        DatasetClass = PinderAlignedDataset
        holo_systems = load_pinder_split(data_dir, split="test", test_setting="holo")
        holo_map = {s["id"]: s for s in holo_systems}
        for s in target_systems:
            if s["id"] in holo_map:
                s["holo_receptor_path"] = holo_map[s["id"]].get("receptor_path")
                s["holo_ligand_path"] = holo_map[s["id"]].get("ligand_path")
                s["holo_receptor_id"] = holo_map[s["id"]].get("receptor_id")
                s["holo_ligand_id"] = holo_map[s["id"]].get("ligand_id")

    from alphasurf.utils.config_utils import merge_surface_config

    on_fly_cfg = getattr(cfg, "on_fly", None)
    mode = "on_fly" if on_fly_cfg is not None else "disk"
    surface_dir = None
    graph_dir = None
    if mode == "disk":
        surface_data_name = getattr(cfg, "surface_data_name", None)
        graph_data_name = getattr(cfg, "graph_data_name", None)
        surface_dir = os.path.join(
            cfg.cfg_surface.data_dir,
            surface_data_name or cfg.cfg_surface.data_name,
        )
        graph_dir = os.path.join(
            cfg.cfg_graph.data_dir,
            graph_data_name or cfg.cfg_graph.data_name,
        )
    esm_dir = getattr(cfg.cfg_graph, "esm_dir", None)

    surface_config = merge_surface_config(cfg.cfg_surface, on_fly_cfg)
    surface_config.use_poisson = "poisson" in cfg.encoder.name
    graph_config = merge_surface_config(cfg.cfg_graph, on_fly_cfg)

    protein_loader = ProteinLoader(
        mode=mode,
        pdb_dir=pdb_dir,
        surface_dir=surface_dir,
        graph_dir=graph_dir,
        esm_dir=esm_dir,
        surface_config=surface_config,
        graph_config=graph_config,
        noise_augmentor=None,
    )

    dataset = DatasetClass(
        systems=target_systems,
        protein_loader=protein_loader,
        pdb_dir=pdb_dir,
        neg_to_pos_ratio=float(cfg.neg_to_pos_ratio),
        max_pos_per_pair=int(cfg.max_pos_per_pair),
        interface_distance_graph=float(cfg.interface_distance_graph),
        interface_distance_surface=float(cfg.interface_distance_surface),
        surface_neg_to_pos_ratio=float(cfg.surface_neg_to_pos_ratio),
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        collate_fn=collate_fn,
        pin_memory=cfg.loader.pin_memory,
    )


def evaluate_setting(
    model, test_loader, device, cfg, test_setting, ckpt_label="provided"
):
    """Run per-system AUROC + balanced accuracy evaluation for one setting.

    Returns dict with keys: auroc_mean, auroc_median, bacc_mean, bacc_median,
    auroc_homo_mean, auroc_hetero_mean, bacc_homo_mean, bacc_hetero_mean, n_systems, ...
    or None if no systems evaluated.
    """
    n_batches = 0
    n_skipped = 0
    system_results = []  # (auroc, bacc, is_homodimer, system_id)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch is None:
                n_skipped += 1
                continue

            batch = batch.to(device)
            if batch.num_graphs < getattr(cfg, "min_batch_size", 2):
                n_skipped += 1
                continue

            if isinstance(batch.label, list):
                per_system_labels = [lab.reshape(-1) for lab in batch.label]
            else:
                per_system_labels = None

            out = model(batch)
            pair_logits = out["graph"]["pair_logit"].squeeze(-1)

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
                        pair_logits[offset : offset + n_pairs]
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()
                    )
                    sys_lab = sys_labels.cpu().numpy().astype(int).ravel()
                    n_pos = sys_lab.sum()
                    n_neg = len(sys_lab) - n_pos
                    sid = sys_ids[si] if si < len(sys_ids) else None
                    if n_pos > 0 and n_neg > 0:
                        sys_auroc = roc_auc_score(sys_lab, sys_logits)
                        sys_preds = (sys_logits > 0).astype(int)
                        sys_bacc = balanced_accuracy_score(sys_lab, sys_preds)
                        is_homo = _is_homodimer(sid) if sid else False
                        system_results.append((sys_auroc, sys_bacc, is_homo, sid))
                    offset += n_pairs

            n_batches += 1

    print(
        f"  {test_setting}: {n_batches} batches, {n_skipped} skipped, {len(system_results)} systems evaluated"
    )

    if not system_results:
        print(f"  {test_setting}: No valid systems evaluated!")
        _dump_results(cfg, ckpt_label, test_setting, None, [])
        return None

    aurocs = np.array([r[0] for r in system_results])
    baccs = np.array([r[1] for r in system_results])
    homo_mask = np.array([r[2] for r in system_results])
    hetero_mask = ~homo_mask

    result = {
        "auroc_mean": aurocs.mean(),
        "auroc_median": float(np.median(aurocs)),
        "auroc_std": aurocs.std(),
        "bacc_mean": baccs.mean(),
        "bacc_median": float(np.median(baccs)),
        "bacc_std": baccs.std(),
        "n_systems": len(system_results),
        "n_homo": int(homo_mask.sum()),
        "n_hetero": int(hetero_mask.sum()),
    }

    if homo_mask.sum() > 0:
        result["auroc_homo_mean"] = float(aurocs[homo_mask].mean())
        result["bacc_homo_mean"] = float(baccs[homo_mask].mean())
    if hetero_mask.sum() > 0:
        result["auroc_hetero_mean"] = float(aurocs[hetero_mask].mean())
        result["bacc_hetero_mean"] = float(baccs[hetero_mask].mean())

    per_system_rows = [
        {
            "system_id": r[3],
            "auroc": r[0],
            "bacc": r[1],
            "is_homodimer": r[2],
        }
        for r in system_results
    ]

    print(
        f"  >>> {test_setting} AUROC: Mean={result['auroc_mean']:.4f}, Median={result['auroc_median']:.4f} <<<"
    )
    print(
        f"  >>> {test_setting} BACC:  Mean={result['bacc_mean']:.4f}, Median={result['bacc_median']:.4f} <<<"
    )
    print(
        f"  >>> {test_setting} Homo/Hetero: {result['n_homo']}/{result['n_hetero']} <<<"
    )
    if result.get("auroc_homo_mean") is not None:
        print(f"  >>> {test_setting} AUROC Homo: {result['auroc_homo_mean']:.4f} <<<")
    if result.get("auroc_hetero_mean") is not None:
        print(
            f"  >>> {test_setting} AUROC Hetero: {result['auroc_hetero_mean']:.4f} <<<"
        )

    _dump_results(cfg, ckpt_label, test_setting, result, per_system_rows)

    return result


def _dump_results(cfg, ckpt_label, test_setting, summary, per_system_rows):
    """Write per-system CSV and append summary row when cfg.dump_per_system is set."""
    dump_dir = getattr(cfg, "dump_per_system", None)
    if not dump_dir:
        return

    import pandas as pd

    os.makedirs(dump_dir, exist_ok=True)
    run_name = getattr(cfg, "run_name", "default")

    if per_system_rows:
        csv_path = os.path.join(dump_dir, f"{run_name}_{ckpt_label}_{test_setting}.csv")
        pd.DataFrame(per_system_rows).to_csv(csv_path, index=False)
        print(
            f"  Dumped per-system results: {csv_path} ({len(per_system_rows)} systems)"
        )

    if summary is not None:
        summary_path = os.path.join(dump_dir, f"{run_name}_summary.csv")
        row = {
            "run_name": run_name,
            "ckpt_label": ckpt_label,
            "test_setting": test_setting,
            **{k: v for k, v in summary.items()},
        }
        df = pd.DataFrame([row])
        if os.path.exists(summary_path):
            df.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            df.to_csv(summary_path, mode="w", header=True, index=False)
        print(f"  Appended summary row: {summary_path}")


def load_model_from_checkpoint(ckpt_path, cfg):
    """Load model from checkpoint using saved training config for correct dims."""
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    train_cfg = ckpt_data.get("hyper_parameters", {}).get("cfg")
    if train_cfg:
        OmegaConf.set_struct(train_cfg, False)
        for key in [
            "test_setting",
            "ckpt_path",
            "device",
            "min_batch_size",
            "on_fly",
            "run_name",
            "dump_per_system",
            "common_only",
            "surface_data_name",
            "graph_data_name",
        ]:
            if key in cfg:
                train_cfg[key] = cfg[key]

        cfg = train_cfg
    model = PinderPairModule.load_from_checkpoint(ckpt_path, cfg=cfg)
    del ckpt_data
    return model, cfg


def _print_summary_table(all_results, ckpt_label):
    """Print a summary table for one checkpoint across all settings."""
    header = f"{ckpt_label:^8} | {'AUROC':^18} | {'BACC':^18} | {'AUROC Homo':^12} | {'AUROC Hetero':^12} | {'N':^7}"
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for setting in ["holo", "apo", "af2"]:
        r = all_results.get(setting)
        if r is None:
            print(
                f"{setting:^8} | {'N/A':^18} | {'N/A':^18} | {'N/A':^12} | {'N/A':^12} | {'0':^7}"
            )
        else:
            auroc_str = f"{r['auroc_mean']:.4f} +/- {r['auroc_std']:.4f}"
            bacc_str = f"{r['bacc_mean']:.4f} +/- {r['bacc_std']:.4f}"
            homo_str = (
                f"{r.get('auroc_homo_mean', float('nan')):.4f}"
                if r.get("n_homo", 0) > 0
                else "N/A"
            )
            hetero_str = (
                f"{r.get('auroc_hetero_mean', float('nan')):.4f}"
                if r.get("n_hetero", 0) > 0
                else "N/A"
            )
            n_str = f"{r['n_homo']}/{r['n_hetero']}"
            print(
                f"{setting:^8} | {auroc_str:^18} | {bacc_str:^18} | {homo_str:^12} | {hetero_str:^12} | {n_str:^7}"
            )
    print(sep)


def run_all_tests(cfg, model=None):
    """Run per-system AUROC evaluation for holo/apo/af2 on best and last checkpoints."""
    ckpt_path = getattr(cfg, "ckpt_path", None)
    if not ckpt_path:
        print("No checkpoint path specified. Skipping tests.")
        return

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cfg.device}")

    ckpts_to_eval = {"provided": ckpt_path}
    last_ckpt = Path(ckpt_path).parent / "last.ckpt"
    if not last_ckpt.exists() and Path(ckpt_path).parent.name.startswith("epoch="):
        last_ckpt = Path(ckpt_path).parent.parent / "last.ckpt"
    if last_ckpt.exists() and str(last_ckpt) != ckpt_path:
        ckpts_to_eval["last"] = str(last_ckpt)

    for ckpt_label, ckpt_file in ckpts_to_eval.items():
        print(f"\n{'=' * 50}")
        print(f"Checkpoint: {ckpt_label} ({ckpt_file})")
        print(f"{'=' * 50}")

        loaded_model, eval_cfg = load_model_from_checkpoint(ckpt_file, cfg)
        loaded_model = loaded_model.to(device)
        loaded_model.eval()

        all_results = {}
        for setting in ["holo", "apo", "af2"]:
            print(f"\n--- {setting} ---")
            try:
                test_loader = setup_test_loader(eval_cfg, setting)
                result = evaluate_setting(
                    loaded_model,
                    test_loader,
                    device,
                    eval_cfg,
                    setting,
                    ckpt_label=ckpt_label,
                )
                all_results[setting] = result
            except Exception as e:
                print(f"  {setting}: FAILED - {e}")

        _print_summary_table(all_results, ckpt_label)
        del loaded_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    pl.seed_everything(cfg.seed, workers=True)

    test_setting = getattr(cfg, "test_setting", "apo")
    ckpt_path = getattr(cfg, "ckpt_path", None)

    print(f"\n{'=' * 60}\nPINDER-PAIR TEST (Setting: {test_setting})\n{'=' * 60}\n")

    if test_setting == "all":
        run_all_tests(cfg)
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{cfg.device}")

        model, eval_cfg = (
            load_model_from_checkpoint(ckpt_path, cfg)
            if ckpt_path
            else (PinderPairModule(cfg), cfg)
        )
        model = model.to(device)
        model.eval()

        ckpt_label = Path(ckpt_path).stem if ckpt_path else "model"
        test_loader = setup_test_loader(eval_cfg, test_setting)
        evaluate_setting(
            model,
            test_loader,
            device,
            eval_cfg,
            test_setting,
            ckpt_label=ckpt_label,
        )


if __name__ == "__main__":
    main()
