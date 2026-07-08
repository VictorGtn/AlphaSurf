"""
Training script for S3F-style pretraining on CATH.

Usage:
    python -m alphasurf.tasks.s3f_pretrain.train \
        device=0 run_name=s3f_pretrain_v1
"""

import faulthandler
import os
import sys
import warnings
from pathlib import Path

faulthandler.enable()

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")

import hydra  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger  # noqa: E402

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.tasks.s3f_pretrain.datamodule import S3FPretrainDataModule  # noqa: E402
from alphasurf.tasks.s3f_pretrain.pl_model import S3FPretrainModule  # noqa: E402
from alphasurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger  # noqa: E402

torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.resolve(cfg)

    pl.seed_everything(cfg.seed, workers=True)

    if getattr(cfg.train, "detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    print(f"\n{'=' * 60}\nS3F PRETRAIN TASK (ON-THE-FLY MODE)\n{'=' * 60}\n")

    datamodule = S3FPretrainDataModule(cfg)
    model = S3FPretrainModule(cfg)

    if os.environ.get("ATOMSURF_VERSION"):
        version_name = os.environ.get("ATOMSURF_VERSION")
    else:
        version = TensorBoardLogger(save_dir=cfg.log_dir).version
        version_name = f"version_{version}_{cfg.run_name}"
    tb_logger = TensorBoardLogger(
        save_dir=cfg.log_dir, version=version_name, name=cfg.run_name
    )

    loggers = [tb_logger]
    if cfg.use_wandb:
        add_wandb_logger(loggers, projectname=cfg.project_name, runname=cfg.run_name)

    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{acc/val:.3f}",
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
        fast_dev_run=getattr(cfg.train, "fast_dev_run", False),
        **params,
    )

    user_ckpt_path = getattr(cfg, "ckpt_path", None)
    ckpt_path = user_ckpt_path

    if os.environ.get("ATOMSURF_RESUME") == "True":
        ckpt_dir = Path(tb_logger.log_dir) / "checkpoints"
        if ckpt_dir.exists():
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
                print(f"Auto-resume from {ckpt_path}")
            else:
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                if ckpts:
                    ckpt_path = str(max(ckpts, key=os.path.getctime))
                    print(f"Auto-resume from {ckpt_path}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    if "--resume" in sys.argv:
        sys.argv.remove("--resume")
        os.environ["ATOMSURF_RESUME"] = "True"
    main()
