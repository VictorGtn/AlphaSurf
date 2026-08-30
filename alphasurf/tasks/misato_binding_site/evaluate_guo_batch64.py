"""Evaluate frame-0 MISATO binding sites with Guo-style batch-64 metrics."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from alphasurf.tasks.misato_binding_site.datamodule import (
    MisatoBindingSiteDataModule,
)
from alphasurf.tasks.misato_binding_site.pl_model import MisatoBindingSiteModule
from alphasurf.tasks.misato_binding_site.tune_threshold import (
    best_validation_threshold,
)


def _batch_values(value, count):
    if isinstance(value, list):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return [value] * count


@torch.inference_mode()
def collect_system_predictions(model, loader, device):
    predictions = {}

    for batch in loader:
        if batch is None:
            continue

        batch = batch.to(device)
        _, logits, _ = model.step(batch)
        if logits is None:
            continue

        batch_labels = batch.y if isinstance(batch.y, list) else [batch.y]
        system_ids = _batch_values(batch.pdb_id, len(batch_labels))
        if len(system_ids) != len(batch_labels):
            raise RuntimeError(
                "MISATO batch system-id count does not match label count: "
                f"{len(system_ids)} != {len(batch_labels)}"
            )

        offset = 0
        for system_id, labels in zip(system_ids, batch_labels):
            labels = labels.detach().cpu().view(-1)
            length = len(labels)
            system_logits = logits[offset : offset + length].detach().cpu()
            if len(system_logits) != length:
                raise RuntimeError(
                    f"MISATO logit/label mismatch for system {system_id}"
                )
            predictions[str(system_id)] = {
                "logits": system_logits,
                "labels": labels,
            }
            offset += length

        if offset != len(logits):
            raise RuntimeError(
                f"MISATO batch logit count mismatch: consumed {offset}, "
                f"received {len(logits)}"
            )

    return predictions


def concatenate_predictions(predictions, system_ids=None):
    if system_ids is None:
        system_ids = list(predictions)
    logits = torch.cat([predictions[system_id]["logits"] for system_id in system_ids])
    labels = (
        torch.cat([predictions[system_id]["labels"] for system_id in system_ids])
        .numpy()
        .astype(np.int64)
    )
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    return labels, probabilities


def guo_batch64_metrics(predictions, threshold, systems_per_batch=64):
    system_ids = list(predictions)
    weighted_sums = {"f1": 0.0, "auroc": 0.0, "auprc": 0.0}
    total_residues = 0
    n_batches = 0

    for start in range(0, len(system_ids), systems_per_batch):
        batch_system_ids = system_ids[start : start + systems_per_batch]
        labels, probabilities = concatenate_predictions(predictions, batch_system_ids)
        n_residues = len(labels)
        metrics = {
            "f1": f1_score(
                labels,
                (probabilities >= threshold).astype(np.int64),
                zero_division=0,
            ),
            "auroc": roc_auc_score(labels, probabilities),
            "auprc": average_precision_score(labels, probabilities),
        }
        for name, value in metrics.items():
            weighted_sums[name] += float(value) * n_residues
        total_residues += n_residues
        n_batches += 1

    return {
        "systems_per_batch": systems_per_batch,
        "n_systems": len(system_ids),
        "n_batches": n_batches,
        "validation_threshold": float(threshold),
        "f1": weighted_sums["f1"] / total_residues,
        "auroc": weighted_sums["auroc"] / total_residues,
        "auprc": weighted_sums["auprc"] / total_residues,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)

    checkpoint_paths = [Path(path) for path in cfg.guo_checkpoint_paths]
    if not checkpoint_paths:
        raise ValueError("Provide +guo_checkpoint_paths=[/path/a.ckpt,...]")
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    with open_dict(cfg):
        cfg.eval_frame_mode = "first"
        cfg.eval_frame_index = 0
        cfg.loader.persistent_workers = False

    pl.seed_everything(int(cfg.seed), workers=True)
    datamodule = MisatoBindingSiteDataModule(cfg)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    results = []

    for checkpoint_path in checkpoint_paths:
        model = MisatoBindingSiteModule.load_from_checkpoint(
            str(checkpoint_path), cfg=cfg, map_location=device
        )
        model.to(device).eval()

        validation_predictions = collect_system_predictions(
            model, datamodule.val_dataloader(), device
        )
        validation_labels, validation_probabilities = concatenate_predictions(
            validation_predictions
        )
        threshold = best_validation_threshold(
            validation_labels, validation_probabilities
        )
        test_predictions = collect_system_predictions(
            model, datamodule.test_dataloader(), device
        )
        metrics = guo_batch64_metrics(test_predictions, threshold)
        result = {
            "checkpoint": str(checkpoint_path),
            "frame_mode": "first",
            "aggregation": "residue-weighted mean over ordered 64-system chunks",
            **metrics,
        }
        results.append(result)
        print(
            f"{checkpoint_path}: threshold={metrics['validation_threshold']:.6f} "
            f"f1={metrics['f1']:.6f} auroc={metrics['auroc']:.6f} "
            f"auprc={metrics['auprc']:.6f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(cfg.guo_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved Guo-style batch-64 results to {output_path}")


if __name__ == "__main__":
    main()
