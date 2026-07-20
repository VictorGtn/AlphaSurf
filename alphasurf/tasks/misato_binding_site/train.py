"""Train AlphaSurf on MISATO residue-level binding-site prediction."""

import faulthandler
import json
import os
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from alphasurf.tasks.misato_binding_site.datamodule import MisatoBindingSiteDataModule
from alphasurf.tasks.misato_binding_site.pl_model import MisatoBindingSiteModule
from alphasurf.utils.callbacks import (
    CommandLoggerCallback,
    GPUMemoryCallback,
    add_wandb_logger,
)

faulthandler.enable()
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy("file_descriptor")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)
    pl.seed_everything(cfg.seed, workers=True)

    datamodule = MisatoBindingSiteDataModule(cfg)
    model = MisatoBindingSiteModule(cfg)
    version = os.environ.get("ATOMSURF_VERSION")
    if version is None:
        version = (
            f"version_{TensorBoardLogger(save_dir=cfg.log_dir).version}_{cfg.run_name}"
        )
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir, version=version, name=cfg.run_name
    )
    loggers = [tb_logger]
    if cfg.use_wandb:
        add_wandb_logger(loggers, projectname=cfg.project_name, runname=cfg.run_name)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{step}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor=cfg.train.to_monitor,
        mode=cfg.train.monitor_mode,
        save_last=True,
        save_top_k=cfg.train.save_top_k,
    )
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        checkpoint_callback,
        CommandLoggerCallback(f"python3 {' '.join(sys.argv)}"),
    ]
    if cfg.train.early_stopping:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor=cfg.train.to_monitor,
                patience=cfg.train.early_stoping_patience,
                mode=cfg.train.monitor_mode,
            )
        )
    if cfg.profile_gpu_memory:
        callbacks.append(GPUMemoryCallback())
    accelerator = (
        {"accelerator": "gpu", "devices": [cfg.device]}
        if torch.cuda.is_available()
        else {}
    )
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=tb_logger.log_dir,
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        val_check_interval=cfg.train.val_check_interval,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        gradient_clip_val=cfg.train.gradient_clip_val,
        fast_dev_run=cfg.train.fast_dev_run,
        **accelerator,
    )

    ckpt_path = cfg.ckpt_path
    if os.environ.get("ATOMSURF_RESUME") == "True":
        # Prefer progress from this run over the original warm-start checkpoint.
        # HPC checkpoints are written on the Slurm preemption signal; last.ckpt
        # is the fallback for ordinary process restarts.
        checkpoint_dir = Path(tb_logger.log_dir) / "checkpoints"
        local_ckpts = list(checkpoint_dir.glob("*.ckpt"))
        local_ckpts.extend(Path(tb_logger.log_dir).glob("hpc_ckpt_*.ckpt"))
        if local_ckpts:
            ckpt_path = str(max(local_ckpts, key=lambda path: path.stat().st_mtime))
            print(f"Auto-resume from run-local checkpoint: {ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    if cfg.test_after_fit and not cfg.train.fast_dev_run:
        reverse = cfg.train.monitor_mode == "max"
        ranked = sorted(
            checkpoint_callback.best_k_models.items(),
            key=lambda item: float(item[1]),
            reverse=reverse,
        )
        test_top_k = int(getattr(cfg, "test_top_k", 4))
        ranked = ranked[:test_top_k]
        if not ranked and checkpoint_callback.best_model_path:
            ranked = [
                (
                    checkpoint_callback.best_model_path,
                    checkpoint_callback.best_model_score,
                )
            ]

        test_items = [
            {
                "label": f"rank_{rank}",
                "rank": rank,
                "path": path,
                "monitor_value": monitor_value,
            }
            for rank, (path, monitor_value) in enumerate(ranked, start=1)
        ]
        if bool(getattr(cfg, "test_last_checkpoint", True)):
            last_path = checkpoint_callback.last_model_path
            if last_path and all(item["path"] != last_path for item in test_items):
                test_items.append(
                    {
                        "label": "last",
                        "rank": None,
                        "path": last_path,
                        "monitor_value": None,
                    }
                )

        test_summary = []
        for index, item in enumerate(test_items, start=1):
            path = item["path"]
            monitor_value = item["monitor_value"]
            print(
                f"\nTesting checkpoint {item['label']} ({index}/{len(test_items)}): "
                f"{path} ({cfg.train.to_monitor}={monitor_value})"
            )
            result = trainer.test(
                model, datamodule=datamodule, ckpt_path=path, verbose=True
            )[0]
            record = {
                "label": item["label"],
                "rank": item["rank"],
                "checkpoint": str(path),
                "monitor": cfg.train.to_monitor,
                "monitor_value": (
                    float(monitor_value) if monitor_value is not None else None
                ),
                "metrics": {key: float(value) for key, value in result.items()},
            }
            test_summary.append(record)
            prefixed = {
                f"checkpoint_{item['label']}/{key}": value
                for key, value in record["metrics"].items()
            }
            for logger in loggers:
                logger.log_metrics(prefixed, step=trainer.global_step)

        summary_path = Path(tb_logger.log_dir) / "top_checkpoint_test_results.json"
        summary_path.write_text(json.dumps(test_summary, indent=2) + "\n")
        print(f"Saved checkpoint test summary to {summary_path}")


if __name__ == "__main__":
    main()
