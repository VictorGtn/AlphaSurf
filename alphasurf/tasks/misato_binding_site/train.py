"""Train AlphaSurf on MISATO residue-level binding-site prediction."""

import faulthandler
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

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{step}",
            dirpath=Path(tb_logger.log_dir) / "checkpoints",
            monitor=cfg.train.to_monitor,
            mode=cfg.train.monitor_mode,
            save_last=True,
            save_top_k=cfg.train.save_top_k,
        ),
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
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    if cfg.test_after_fit and not cfg.train.fast_dev_run:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
