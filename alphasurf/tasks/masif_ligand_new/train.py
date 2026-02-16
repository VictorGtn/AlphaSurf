"""
Training script for MaSIF-Ligand task.

Supports both disk-based and on-the-fly data loading with noise augmentation.
"""

import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.tasks.masif_ligand_new.datamodule import MasifLigandDataModule
from alphasurf.tasks.masif_ligand_new.pl_model import MasifLigandModule
from alphasurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger

torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    pl.seed_everything(cfg.seed, workers=True)

    # Print mode info
    on_fly = getattr(cfg, "on_fly", None)
    if on_fly is not None:
        print("\n" + "=" * 60)
        print("ON-THE-FLY MODE")
        print(f"  Surface method: {getattr(on_fly, 'surface_method', 'msms')}")
        print(f"  Noise mode: {getattr(on_fly, 'noise_mode', 'none')}")
        print("=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("DISK MODE (precomputed surfaces/graphs)")
        print("=" * 60 + "\n")

    # DataModule
    datamodule = MasifLigandDataModule(cfg)

    # Model
    model = MasifLigandModule(cfg)

    # Loggers
    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    if cfg.use_wandb:
        data_dir_name = Path(cfg.data_dir).name if cfg.data_dir else "default"
        mode_suffix = "onfly" if on_fly else "disk"
        wandb_name = f"{cfg.run_name}_{data_dir_name}_{mode_suffix}"
        project = getattr(cfg, "project_name", "masif_ligand")
        add_wandb_logger(loggers, projectname=project, runname=wandb_name)

    # Callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{accuracy_balanced/val:.2f}",
            dirpath=Path(tb_logger.log_dir) / "checkpoints",
            monitor=cfg.train.to_monitor,
            mode="max",
            save_last=True,
            save_top_k=cfg.train.save_top_k,
            verbose=False,
        ),
        pl.callbacks.EarlyStopping(
            monitor=cfg.train.to_monitor,
            patience=cfg.train.early_stoping_patience,
            mode="max",
        ),
        CommandLoggerCallback(command),
    ]

    # Trainer
    params = {}
    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [cfg.device]}

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        val_check_interval=cfg.train.val_check_interval,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        log_every_n_steps=cfg.train.log_every_n_steps,
        max_steps=cfg.train.max_steps,
        gradient_clip_val=cfg.train.gradient_clip_val,
        detect_anomaly=cfg.train.detect_anomaly,
        overfit_batches=cfg.train.overfit_batches,
        **params,
    )

    # Train
    ckpt_path = getattr(cfg, "ckpt_path", None)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Test
    print("=" * 40 + " TEST BEST " + "=" * 40)
    results = trainer.test(model, ckpt_path="best", datamodule=datamodule)
    acc_best = results[0]["accuracy_balanced/test"]
    trainer.logger.log_metrics({"accuracy_balanced/test_best": acc_best})

    print("=" * 40 + " TEST LAST " + "=" * 40)
    results = trainer.test(model, ckpt_path="last", datamodule=datamodule)
    acc_last = results[0]["accuracy_balanced/test"]
    trainer.logger.log_metrics({"accuracy_balanced/test_last": acc_last})


if __name__ == "__main__":
    main()
