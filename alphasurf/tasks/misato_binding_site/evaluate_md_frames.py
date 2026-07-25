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


def frame_spec(mode, *, fraction=None, index=None):
    spec = {"mode": str(mode)}
    if fraction is not None:
        spec["fraction"] = float(fraction)
    if index is not None:
        spec["index"] = int(index)
    return spec


def frame_spec_key(spec):
    return (
        spec["mode"],
        spec.get("fraction"),
        spec.get("index"),
    )


def frame_spec_label(spec):
    if spec["mode"] == "fraction":
        return f"fraction={spec['fraction']:.3f}"
    if spec["mode"] in {"first", "fixed"}:
        return f"fixed-index={spec.get('index', 0)}"
    return spec["mode"]


def training_frame_calibration_specs(training_cfg, random_fractions):
    """Return deterministic validation frames matching the training regime."""
    mode = str(OmegaConf.select(training_cfg, "train_frame_mode", default="first"))
    if mode == "random":
        specs = [
            frame_spec("fraction", fraction=fraction) for fraction in random_fractions
        ]
        protocol = "deterministic uniform grid approximating random-frame training"
    elif mode in {"first", "fixed"}:
        index = int(OmegaConf.select(training_cfg, "train_frame_index", default=0) or 0)
        specs = [frame_spec("fixed", index=index)]
        protocol = "fixed training frame"
    elif mode == "middle":
        specs = [frame_spec("middle")]
        protocol = "middle training frame"
    elif mode == "fraction":
        fraction = float(
            OmegaConf.select(training_cfg, "train_frame_fraction", default=0.5)
        )
        specs = [frame_spec("fraction", fraction=fraction)]
        protocol = "fractional training frame"
    else:
        raise ValueError(f"Unsupported checkpoint training frame mode: {mode}")
    return {
        "training_frame_mode": mode,
        "protocol": protocol,
        "frame_specs": specs,
    }


def checkpoint_training_cfg(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_cfg = checkpoint.get("hyper_parameters", {}).get("cfg")
    if training_cfg is None:
        raise KeyError(f"{checkpoint_path} does not contain hyper_parameters.cfg")
    return training_cfg


def collect_split_frames(model, datamodule, split, specs, device):
    collected = []
    for spec in specs:
        with open_dict(datamodule.cfg):
            datamodule.cfg.eval_frame_mode = spec["mode"]
            if "fraction" in spec:
                datamodule.cfg.eval_frame_fraction = float(spec["fraction"])
            if "index" in spec:
                datamodule.cfg.eval_frame_index = int(spec["index"])
        loader = (
            datamodule.val_dataloader()
            if split == "validation"
            else datamodule.test_dataloader()
        )
        logits, labels, probabilities = collect_predictions(model, loader, device)
        collected.append(
            {
                "spec": dict(spec),
                "logits": logits,
                "labels": labels,
                "probabilities": probabilities,
            }
        )
        print(f"Collected {split} frame {frame_spec_label(spec)}")
    return collected


def short_checkpoint_name(checkpoint):
    path = Path(checkpoint)
    if len(path.parents) >= 3:
        return f"{path.parents[2].name}/{path.parents[1].name}"
    return path.name


def print_recap_table(results):
    headers = [
        "Model",
        "Calibration",
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "AUROC",
        "AUPRC",
    ]
    rows = []
    for result in results:
        groups = [
            (
                "source-frozen",
                result["source_calibration"]["selected_threshold"],
                result["test_pooled"]["threshold_frozen_from_source_validation"],
            ),
            (
                "target-tuned",
                result["target_calibration"]["selected_threshold"],
                result["test_pooled"]["threshold_tuned_on_target_validation"],
            ),
        ]
        for calibration, threshold, metrics in groups:
            rows.append(
                [
                    short_checkpoint_name(result["checkpoint"]),
                    calibration,
                    f"{threshold:.6f}",
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['precision']:.6f}",
                    f"{metrics['recall']:.6f}",
                    f"{metrics['f1']:.6f}",
                    f"{metrics['auroc']:.6f}",
                    f"{metrics['auprc']:.6f}",
                ]
            )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)
    print("\nMD EVALUATION RECAP (pooled residue-level test metrics)")
    print(
        " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    print(
        "\nsource-frozen: threshold selected on validation frames matching "
        "the checkpoint's training regime, then frozen on target test frames."
    )
    print(
        "target-tuned: threshold selected on validation frames matching the "
        "target evaluation frames, then frozen on target test frames."
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_paths = [Path(path) for path in cfg.md_eval_checkpoint_paths]
    fractions = sorted({float(value) for value in cfg.md_eval_frame_fractions})
    random_calibration_fractions = sorted(
        {
            float(value)
            for value in OmegaConf.select(
                cfg,
                "md_eval_random_calibration_frame_fractions",
                default=np.linspace(0.0, 1.0, 11).tolist(),
            )
        }
    )
    forced_source_fractions_raw = OmegaConf.select(
        cfg,
        "md_eval_source_calibration_frame_fractions",
        default=None,
    )
    forced_source_fractions = (
        sorted({float(value) for value in forced_source_fractions_raw})
        if forced_source_fractions_raw is not None
        else None
    )
    if not checkpoint_paths:
        raise ValueError("Provide +md_eval_checkpoint_paths=[/path/a.ckpt,...]")
    if not fractions or any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("MD evaluation frame fractions must lie in [0, 1]")
    if not random_calibration_fractions or any(
        not 0.0 <= value <= 1.0 for value in random_calibration_fractions
    ):
        raise ValueError("Random calibration frame fractions must lie in [0, 1]")
    if forced_source_fractions is not None and (
        not forced_source_fractions
        or any(not 0.0 <= value <= 1.0 for value in forced_source_fractions)
    ):
        raise ValueError("Forced source calibration fractions must lie in [0, 1]")
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    with open_dict(cfg):
        cfg.loader.persistent_workers = False

    datamodule = MisatoBindingSiteDataModule(cfg)
    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    all_results = []
    target_specs = [frame_spec("fraction", fraction=fraction) for fraction in fractions]

    for checkpoint_path in checkpoint_paths:
        print(f"\nLoading {checkpoint_path}")
        training_cfg = checkpoint_training_cfg(checkpoint_path)
        if forced_source_fractions is None:
            source_calibration = training_frame_calibration_specs(
                training_cfg, random_calibration_fractions
            )
        else:
            source_calibration = {
                "training_frame_mode": str(
                    OmegaConf.select(
                        training_cfg, "train_frame_mode", default="unknown"
                    )
                ),
                "protocol": "explicit shared source-validation frame fractions",
                "frame_specs": [
                    frame_spec("fraction", fraction=fraction)
                    for fraction in forced_source_fractions
                ],
            }
        source_specs = source_calibration["frame_specs"]
        model = MisatoBindingSiteModule.load_from_checkpoint(
            str(checkpoint_path), cfg=cfg, map_location=device
        )
        model.to(device).eval()

        validation_specs = []
        seen_specs = set()
        for spec in source_specs + target_specs:
            key = frame_spec_key(spec)
            if key not in seen_specs:
                seen_specs.add(key)
                validation_specs.append(spec)
        validation_frames = collect_split_frames(
            model, datamodule, "validation", validation_specs, device
        )
        validation_by_spec = {
            frame_spec_key(frame["spec"]): frame for frame in validation_frames
        }
        source_validation_frames = [
            validation_by_spec[frame_spec_key(spec)] for spec in source_specs
        ]
        target_validation_frames = [
            validation_by_spec[frame_spec_key(spec)] for spec in target_specs
        ]
        source_val_logits, source_val_labels, source_val_probabilities = (
            concatenate_frame_predictions(source_validation_frames)
        )
        target_val_logits, target_val_labels, target_val_probabilities = (
            concatenate_frame_predictions(target_validation_frames)
        )
        source_threshold = best_validation_threshold(
            source_val_labels, source_val_probabilities
        )
        target_threshold = best_validation_threshold(
            target_val_labels, target_val_probabilities
        )

        test_frames = collect_split_frames(
            model, datamodule, "test", target_specs, device
        )
        test_logits, test_labels, test_probabilities = concatenate_frame_predictions(
            test_frames
        )

        per_frame = []
        for frame in test_frames:
            per_frame.append(
                {
                    "frame_fraction": frame["spec"]["fraction"],
                    "threshold_0.5": metrics_at_threshold(
                        frame["logits"],
                        frame["labels"],
                        frame["probabilities"],
                        0.5,
                    ),
                    "threshold_frozen_from_source_validation": metrics_at_threshold(
                        frame["logits"],
                        frame["labels"],
                        frame["probabilities"],
                        source_threshold,
                    ),
                    "threshold_tuned_on_target_validation": metrics_at_threshold(
                        frame["logits"],
                        frame["labels"],
                        frame["probabilities"],
                        target_threshold,
                    ),
                    # Backward-compatible alias for earlier result consumers.
                    "threshold_tuned_on_multiframe_validation": metrics_at_threshold(
                        frame["logits"],
                        frame["labels"],
                        frame["probabilities"],
                        target_threshold,
                    ),
                }
            )

        result = {
            "checkpoint": str(checkpoint_path),
            "labels": "fixed frame-0 ligand contacts",
            "frame_fractions": fractions,
            "threshold_selection": {
                "source_frozen": "maximum pooled source-validation F1",
                "target_tuned": "maximum pooled target-frame-validation F1",
            },
            # Backward-compatible alias: historically this was target-tuned.
            "selected_threshold": target_threshold,
            "source_calibration": {
                **source_calibration,
                "selected_threshold": source_threshold,
            },
            "target_calibration": {
                "protocol": "target evaluation frames",
                "frame_specs": target_specs,
                "selected_threshold": target_threshold,
            },
            "validation_pooled": {
                "source": {
                    "threshold_0.5": metrics_at_threshold(
                        source_val_logits,
                        source_val_labels,
                        source_val_probabilities,
                        0.5,
                    ),
                    "threshold_tuned": metrics_at_threshold(
                        source_val_logits,
                        source_val_labels,
                        source_val_probabilities,
                        source_threshold,
                    ),
                },
                "target": {
                    "threshold_0.5": metrics_at_threshold(
                        target_val_logits,
                        target_val_labels,
                        target_val_probabilities,
                        0.5,
                    ),
                    "threshold_tuned": metrics_at_threshold(
                        target_val_logits,
                        target_val_labels,
                        target_val_probabilities,
                        target_threshold,
                    ),
                },
            },
            "test_pooled": {
                "threshold_0.5": metrics_at_threshold(
                    test_logits, test_labels, test_probabilities, 0.5
                ),
                "threshold_frozen_from_source_validation": metrics_at_threshold(
                    test_logits,
                    test_labels,
                    test_probabilities,
                    source_threshold,
                ),
                "threshold_tuned_on_target_validation": metrics_at_threshold(
                    test_logits,
                    test_labels,
                    test_probabilities,
                    target_threshold,
                ),
                # Backward-compatible alias for earlier result consumers.
                "threshold_tuned_on_multiframe_validation": metrics_at_threshold(
                    test_logits,
                    test_labels,
                    test_probabilities,
                    target_threshold,
                ),
            },
            "test_per_frame": per_frame,
            "test_frame_macro": {
                "threshold_0.5": macro_frame_summary(per_frame, "threshold_0.5"),
                "threshold_frozen_from_source_validation": macro_frame_summary(
                    per_frame, "threshold_frozen_from_source_validation"
                ),
                "threshold_tuned_on_target_validation": macro_frame_summary(
                    per_frame, "threshold_tuned_on_target_validation"
                ),
                "threshold_tuned_on_multiframe_validation": macro_frame_summary(
                    per_frame, "threshold_tuned_on_multiframe_validation"
                ),
            },
        }
        all_results.append(result)
        source_metrics = result["test_pooled"][
            "threshold_frozen_from_source_validation"
        ]
        target_metrics = result["test_pooled"]["threshold_tuned_on_target_validation"]
        print(
            f"source_threshold={source_threshold:.6f} "
            f"source_frozen_test_f1={source_metrics['f1']:.6f}; "
            f"target_threshold={target_threshold:.6f} "
            f"target_tuned_test_f1={target_metrics['f1']:.6f}; "
            f"auprc={target_metrics['auprc']:.6f} "
            f"auroc={target_metrics['auroc']:.6f}"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_path = Path(cfg.md_eval_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"Saved multi-frame evaluation to {output_path}")
    print_recap_table(all_results)


if __name__ == "__main__":
    main()
