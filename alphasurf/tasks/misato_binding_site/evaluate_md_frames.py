"""Evaluate fixed frame-0 binding labels over deterministic MD conformations."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict

from alphasurf.tasks.misato_binding_site.datamodule import (
    MisatoBindingSiteDataModule,
)
from alphasurf.tasks.misato_binding_site.pl_model import MisatoBindingSiteModule
from alphasurf.tasks.misato_binding_site.tune_threshold import (
    best_validation_threshold,
    collect_predictions,
    metrics_at_threshold,
)


def concatenate_frame_predictions(frame_predictions):
    logits = torch.cat([item["logits"] for item in frame_predictions])
    labels = np.concatenate([item["labels"] for item in frame_predictions])
    probabilities = np.concatenate(
        [item["probabilities"] for item in frame_predictions]
    )
    return logits, labels, probabilities


def macro_frame_summary(frame_results, metric_group):
    metric_names = list(frame_results[0][metric_group])
    summary = {}
    for metric_name in metric_names:
        if metric_name == "threshold":
            continue
        values = np.asarray(
            [frame[metric_group][metric_name] for frame in frame_results],
            dtype=np.float64,
        )
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary


def collect_split_frames(model, datamodule, split, fractions, device):
    collected = []
    for fraction in fractions:
        with open_dict(datamodule.cfg):
            datamodule.cfg.eval_frame_mode = "fraction"
            datamodule.cfg.eval_frame_fraction = float(fraction)
        loader = (
            datamodule.val_dataloader()
            if split == "validation"
            else datamodule.test_dataloader()
        )
        logits, labels, probabilities = collect_predictions(model, loader, device)
        collected.append(
            {
                "fraction": float(fraction),
                "logits": logits,
                "labels": labels,
                "probabilities": probabilities,
            }
        )
        print(f"Collected {split} frame fraction {fraction:.3f}")
    return collected


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_paths = [Path(path) for path in cfg.md_eval_checkpoint_paths]
    fractions = sorted({float(value) for value in cfg.md_eval_frame_fractions})
    if not checkpoint_paths:
        raise ValueError("Provide +md_eval_checkpoint_paths=[/path/a.ckpt,...]")
    if not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("MD evaluation frame fractions must lie in [0, 1]")
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    with open_dict(cfg):
        cfg.loader.persistent_workers = False

    datamodule = MisatoBindingSiteDataModule(cfg)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    all_results = []

    for checkpoint_path in checkpoint_paths:
        print(f"\nLoading {checkpoint_path}")
        model = MisatoBindingSiteModule.load_from_checkpoint(
            str(checkpoint_path), cfg=cfg, map_location=device
        )
        model.to(device).eval()

        validation_frames = collect_split_frames(
            model, datamodule, "validation", fractions, device
        )
        val_logits, val_labels, val_probabilities = concatenate_frame_predictions(
            validation_frames
        )
        tuned_threshold = best_validation_threshold(val_labels, val_probabilities)

        test_frames = collect_split_frames(model, datamodule, "test", fractions, device)
        test_logits, test_labels, test_probabilities = concatenate_frame_predictions(
            test_frames
        )

        per_frame = []
        for frame in test_frames:
            per_frame.append(
                {
                    "frame_fraction": frame["fraction"],
                    "threshold_0.5": metrics_at_threshold(
                        frame["logits"],
                        frame["labels"],
                        frame["probabilities"],
                        0.5,
                    ),
                    "threshold_tuned_on_multiframe_validation": metrics_at_threshold(
                        frame["logits"],
                        frame["labels"],
                        frame["probabilities"],
                        tuned_threshold,
                    ),
                }
            )

        result = {
            "checkpoint": str(checkpoint_path),
            "labels": "fixed frame-0 ligand contacts",
            "frame_fractions": fractions,
            "threshold_selection": "maximum pooled multi-frame validation F1",
            "selected_threshold": tuned_threshold,
            "validation_pooled": {
                "threshold_0.5": metrics_at_threshold(
                    val_logits, val_labels, val_probabilities, 0.5
                ),
                "threshold_tuned": metrics_at_threshold(
                    val_logits, val_labels, val_probabilities, tuned_threshold
                ),
            },
            "test_pooled": {
                "threshold_0.5": metrics_at_threshold(
                    test_logits, test_labels, test_probabilities, 0.5
                ),
                "threshold_tuned_on_multiframe_validation": metrics_at_threshold(
                    test_logits, test_labels, test_probabilities, tuned_threshold
                ),
            },
            "test_per_frame": per_frame,
            "test_frame_macro": {
                "threshold_0.5": macro_frame_summary(per_frame, "threshold_0.5"),
                "threshold_tuned_on_multiframe_validation": macro_frame_summary(
                    per_frame, "threshold_tuned_on_multiframe_validation"
                ),
            },
        }
        all_results.append(result)
        tuned = result["test_pooled"]["threshold_tuned_on_multiframe_validation"]
        print(
            f"threshold={tuned_threshold:.6f} pooled_test_f1={tuned['f1']:.6f} "
            f"auprc={tuned['auprc']:.6f} auroc={tuned['auroc']:.6f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(cfg.md_eval_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"Saved multi-frame evaluation to {output_path}")


if __name__ == "__main__":
    main()
