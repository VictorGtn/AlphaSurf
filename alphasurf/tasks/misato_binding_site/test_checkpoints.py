"""Evaluate every retained MISATO binding-site checkpoint in one directory."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from alphasurf.tasks.misato_binding_site.datamodule import (
    MisatoBindingSiteDataModule,
)
from alphasurf.tasks.misato_binding_site.pl_model import MisatoBindingSiteModule


def _monitor_score(checkpoint, path):
    """Read this checkpoint's validation score from ModelCheckpoint state."""
    for state in checkpoint.get("callbacks", {}).values():
        if not isinstance(state, dict):
            continue
        for saved_path, score in state.get("best_k_models", {}).items():
            if Path(saved_path).name == path.name:
                return float(score)
    return None


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    checkpoint_dir = Path(cfg.checkpoint_dir)
    epoch_paths = sorted(checkpoint_dir.glob("epoch=*.ckpt"))
    last_paths = sorted(
        checkpoint_dir.glob("last*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not epoch_paths and not last_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    scored = []
    for path in epoch_paths:
        checkpoint = torch.load(path, map_location="cpu")
        scored.append(
            {
                "path": path,
                "score": _monitor_score(checkpoint, path),
                "epoch": int(checkpoint.get("epoch", -1)),
                "global_step": int(checkpoint.get("global_step", -1)),
            }
        )
        del checkpoint

    reverse = cfg.train.monitor_mode == "max"
    ranked = [item for item in scored if item["score"] is not None]
    ranked.sort(key=lambda item: item["score"], reverse=reverse)
    unranked = [item for item in scored if item["score"] is None]
    selected = (ranked + unranked)[: int(getattr(cfg, "test_top_k", 4))]
    for rank, item in enumerate(selected, start=1):
        item["label"] = f"rank_{rank}"
        item["rank"] = rank

    if bool(getattr(cfg, "include_last_checkpoint", True)) and last_paths:
        last_path = last_paths[0]
        if all(item["path"] != last_path for item in selected):
            checkpoint = torch.load(last_path, map_location="cpu")
            selected.append(
                {
                    "label": "last",
                    "rank": None,
                    "path": last_path,
                    "score": None,
                    "epoch": int(checkpoint.get("epoch", -1)),
                    "global_step": int(checkpoint.get("global_step", -1)),
                }
            )
            del checkpoint

    datamodule = MisatoBindingSiteDataModule(cfg)
    model = MisatoBindingSiteModule(cfg)
    accelerator = (
        {"accelerator": "gpu", "devices": [cfg.device]}
        if torch.cuda.is_available()
        else {}
    )
    trainer = pl.Trainer(logger=False, **accelerator)

    summary = []
    for index, item in enumerate(selected, start=1):
        print(
            f"\nTesting checkpoint {item['label']} ({index}/{len(selected)}): "
            f"{item['path']} "
            f"(epoch={item['epoch']}, {cfg.train.to_monitor}={item['score']})"
        )
        metrics = trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=str(item["path"]),
            verbose=True,
        )[0]
        summary.append(
            {
                "label": item["label"],
                "rank": item["rank"],
                "checkpoint": str(item["path"]),
                "epoch": item["epoch"],
                "global_step": item["global_step"],
                "monitor": cfg.train.to_monitor,
                "monitor_value": item["score"],
                "metrics": {key: float(value) for key, value in metrics.items()},
            }
        )

    configured_output = getattr(cfg, "checkpoint_test_output", None)
    output_path = (
        Path(configured_output)
        if configured_output
        else checkpoint_dir / "top_checkpoint_test_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved checkpoint test summary to {output_path}")


if __name__ == "__main__":
    main()
