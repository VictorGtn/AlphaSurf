"""Tune a residue decision threshold on validation and evaluate it on test."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from alphasurf.tasks.misato_binding_site.datamodule import (
    MisatoBindingSiteDataModule,
)
from alphasurf.tasks.misato_binding_site.pl_model import MisatoBindingSiteModule


@torch.inference_mode()
def collect_predictions(model, loader, device):
    logits_parts = []
    label_parts = []
    for batch in loader:
        if batch is None:
            continue
        batch = batch.to(device)
        _, logits, labels = model.step(batch)
        if logits is None:
            continue
        logits_parts.append(logits.detach().cpu())
        label_parts.append(labels.detach().cpu())
    if not logits_parts:
        raise RuntimeError("No valid batches were produced")
    logits = torch.cat(logits_parts)
    labels = torch.cat(label_parts).long()
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    return logits, labels.numpy().astype(np.int64), probabilities


def metrics_at_threshold(logits, labels, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(np.int64)
    metrics = {
        "threshold": float(threshold),
        "loss": float(F.cross_entropy(logits, torch.from_numpy(labels)).item()),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "positive_prediction_rate": float(predictions.mean()),
    }
    if np.unique(labels).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["auprc"] = float(average_precision_score(labels, probabilities))
    return metrics


def best_validation_threshold(labels, probabilities):
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_f1 = np.max(f1_values)
    candidates = np.flatnonzero(np.isclose(f1_values, best_f1, rtol=0, atol=1e-12))
    # Deterministic tie-break: prefer the candidate closest to the conventional
    # 0.5 threshold.
    index = candidates[np.argmin(np.abs(thresholds[candidates] - 0.5))]
    return float(thresholds[index])


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_paths = [Path(path) for path in cfg.threshold_checkpoint_paths]
    if not checkpoint_paths:
        raise ValueError("Provide +threshold_checkpoint_paths=[/path/a.ckpt,...]")
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    with open_dict(cfg):
        cfg.loader.persistent_workers = False

    datamodule = MisatoBindingSiteDataModule(cfg)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    results = []

    for checkpoint_path in checkpoint_paths:
        print(f"\nLoading {checkpoint_path}")
        model = MisatoBindingSiteModule.load_from_checkpoint(
            str(checkpoint_path), cfg=cfg, map_location=device
        )
        model.to(device).eval()

        val_logits, val_labels, val_probabilities = collect_predictions(
            model, datamodule.val_dataloader(), device
        )
        threshold = best_validation_threshold(val_labels, val_probabilities)
        test_logits, test_labels, test_probabilities = collect_predictions(
            model, datamodule.test_dataloader(), device
        )

        result = {
            "checkpoint": str(checkpoint_path),
            "selection": "maximum pooled validation F1",
            "selected_threshold": threshold,
            "counts": {
                "validation_residues": int(len(val_labels)),
                "validation_positives": int(val_labels.sum()),
                "test_residues": int(len(test_labels)),
                "test_positives": int(test_labels.sum()),
            },
            "validation": {
                "threshold_0.5": metrics_at_threshold(
                    val_logits, val_labels, val_probabilities, 0.5
                ),
                "threshold_tuned": metrics_at_threshold(
                    val_logits, val_labels, val_probabilities, threshold
                ),
            },
            "test": {
                "threshold_0.5": metrics_at_threshold(
                    test_logits, test_labels, test_probabilities, 0.5
                ),
                "threshold_tuned_on_validation": metrics_at_threshold(
                    test_logits, test_labels, test_probabilities, threshold
                ),
            },
        }
        results.append(result)
        tuned = result["test"]["threshold_tuned_on_validation"]
        print(
            f"threshold={threshold:.6f} test_f1={tuned['f1']:.6f} "
            f"precision={tuned['precision']:.6f} recall={tuned['recall']:.6f}"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(cfg.threshold_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved threshold-tuning results to {output_path}")


if __name__ == "__main__":
    main()
