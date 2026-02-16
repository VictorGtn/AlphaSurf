"""
Training script for PINDER-Pair protein-protein interaction task.
"""

import faulthandler
import os
import sys
import warnings
from pathlib import Path

faulthandler.enable()

# Enable segfault debugging to catch which C++ function crashes
CRASH_DEBUG_DIR = os.environ.get("CRASH_DEBUG_DIR")
if CRASH_DEBUG_DIR:
    from alphasurf.utils.segfault_debugger import enable_debugging

    enable_debugging(CRASH_DEBUG_DIR)

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
warnings.filterwarnings("ignore")

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.tasks.pinder_pair.datamodule import PinderPairDataModule
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule
from alphasurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger

torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    command = f"python3 {' '.join(sys.argv)}"
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    pl.seed_everything(cfg.seed, workers=True)

    # DEBUG: Enable anomaly detection (slows training ~2x)
    if getattr(cfg.train, "detect_anomaly", False):
        torch.autograd.set_detect_anomaly(True)

    # Print mode info
    on_fly = getattr(cfg, "on_fly", None)
    mode_str = "ON-THE-FLY" if on_fly else "DISK"
    print(f"\n{'=' * 60}\nPINDER-PAIR TASK ({mode_str} MODE)\n{'=' * 60}\n")

    # DataModule
    datamodule = PinderPairDataModule(cfg)

    # Model
    model = PinderPairModule(cfg)

    # Loggers
    if os.environ.get("ATOMSURF_VERSION"):
        # Force a specific version name (e.g. SLURM_JOB_ID) to simplify restarts
        version_name = os.environ.get("ATOMSURF_VERSION")
        tb_logger = TensorBoardLogger(
            save_dir=cfg.log_dir, version=version_name, name=cfg.run_name
        )
    else:
        # Default behavior: auto-increment version
        version = TensorBoardLogger(save_dir=cfg.log_dir).version
        version_name = f"version_{version}_{cfg.run_name}"
        tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)

    loggers = [tb_logger]

    if cfg.use_wandb:
        project = getattr(cfg, "project_name", "pinder_pair")
        add_wandb_logger(loggers, projectname=project, runname=cfg.run_name)

    # Callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.ModelCheckpoint(
            filename="{epoch}-{auroc/val:.3f}",
            dirpath=Path(tb_logger.log_dir) / "checkpoints",
            monitor=cfg.train.to_monitor,
            mode="max",
            save_last=True,
            save_top_k=cfg.train.save_top_k,
            verbose=False,
        ),
        # Additional frequent checkpointing for crash resilience
        pl.callbacks.ModelCheckpoint(
            filename="step_{step:06d}",
            dirpath=Path(tb_logger.log_dir) / "checkpoints",
            every_n_train_steps=100,
            save_top_k=-1,
            save_last=True,
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
    # Auto-resume logic
    user_ckpt_path = getattr(cfg, "ckpt_path", None)
    ckpt_path = user_ckpt_path

    if os.environ.get("ATOMSURF_RESUME") == "True":
        # Check in the current log directory for checkpoints
        ckpt_dir = Path(tb_logger.log_dir) / "checkpoints"

        found_local = False
        if ckpt_dir.exists():
            # Prioritize 'last.ckpt'
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.exists():
                ckpt_path = str(last_ckpt)
                print(
                    f"Auto-resume: Found local progress, ignoring user ckpt. Resuming from {ckpt_path}"
                )
                found_local = True
            else:
                # Fallback to most recent in current dir
                ckpts = list(ckpt_dir.glob("*.ckpt"))
                if ckpts:
                    ckpt_path = str(max(ckpts, key=os.path.getctime))
                    print(
                        f"Auto-resume: Found local progress, ignoring user ckpt. Resuming from {ckpt_path}"
                    )
                    found_local = True

        if not found_local:
            if ckpt_path is not None:
                print(
                    f"No local progress found. Starting from user-provided checkpoint: {ckpt_path}"
                )
            else:
                print(
                    f"No checkpoins found in {ckpt_dir} and no user ckpt provided. Starting from scratch."
                )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Test
    # Determine test settings to run
    test_setting = getattr(cfg, "test_setting", "apo")
    if test_setting == "all":
        test_settings = ["holo", "apo", "af2"]
    else:
        test_settings = [test_setting]

    print(f"\nRunning tests for settings: {test_settings}")

    for setting in test_settings:
        print(f"\n{'=' * 40} TEST SETTING: {setting} {'=' * 40}")

        # Update config and recreate datamodule for this setting
        OmegaConf.set_struct(cfg, False)
        cfg.test_setting = setting
        # Disable persistent workers for testing to avoid crash
        # persistent_workers often causes issues on resume/re-init with KeOps
        if hasattr(cfg, "loader"):
            cfg.loader.persistent_workers = False
        OmegaConf.set_struct(cfg, True)

        # Re-initialize DataModule with new setting
        test_datamodule = PinderPairDataModule(cfg)

        # Test Best
        print(f"--- Testing BEST checkpoint ({setting}) ---")
        results = trainer.test(model, ckpt_path="best", datamodule=test_datamodule)
        auroc_best = results[0]["auroc/test"]
        trainer.logger.log_metrics({f"auroc/test_best_{setting}": auroc_best})

        # Test Last
        print(f"--- Testing LAST checkpoint ({setting}) ---")
        results = trainer.test(model, ckpt_path="last", datamodule=test_datamodule)
        auroc_last = results[0]["auroc/test"]
        trainer.logger.log_metrics({f"auroc/test_last_{setting}": auroc_last})


if __name__ == "__main__":
    # Handle --resume flag manually to avoid Hydra conflict
    if "--resume" in sys.argv:
        sys.argv.remove("--resume")
        os.environ["ATOMSURF_RESUME"] = "True"
    main()
