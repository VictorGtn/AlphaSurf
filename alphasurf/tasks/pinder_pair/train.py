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

import hydra  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger  # noqa: E402

torch.multiprocessing.set_sharing_strategy("file_system")

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.tasks.pinder_pair.datamodule import PinderPairDataModule  # noqa: E402
from alphasurf.tasks.pinder_pair.pl_model import PinderPairModule  # noqa: E402
from alphasurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger  # noqa: E402

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

    # Test: per-system AUROC for holo/apo/af2 on best and last checkpoints
    print(f"\n{'=' * 60}\nPOST-TRAINING EVALUATION\n{'=' * 60}")

    # Disable persistent workers for testing to avoid crash
    OmegaConf.set_struct(cfg, False)
    if hasattr(cfg, "loader"):
        cfg.loader.persistent_workers = False
    OmegaConf.set_struct(cfg, True)

    for ckpt_label in ["best", "last"]:
        print(f"\n{'=' * 50}\nCheckpoint: {ckpt_label}\n{'=' * 50}")

        ckpt_results = {}
        for setting in ["holo", "apo", "af2"]:
            print(f"\n--- {setting} ---")
            OmegaConf.set_struct(cfg, False)
            cfg.test_setting = setting
            OmegaConf.set_struct(cfg, True)

            test_datamodule = PinderPairDataModule(cfg)
            results = trainer.test(
                model, ckpt_path=ckpt_label, datamodule=test_datamodule
            )

            if results:
                r = results[0]
                mean = r.get("auroc/test_mean", r.get("auroc/test", 0))
                median = r.get("auroc/test_median", 0)
                n_sys = r.get("auroc/test_n_systems", 0)
                bacc_mean = r.get("bacc/test_mean", 0)
                trainer.logger.log_metrics(
                    {
                        f"test_{ckpt_label}_{setting}_auroc_mean": mean,
                        f"test_{ckpt_label}_{setting}_auroc_median": median,
                        f"test_{ckpt_label}_{setting}_bacc_mean": bacc_mean,
                    }
                )
                print(f"  AUROC: mean={mean:.4f}, median={median:.4f}, n={int(n_sys)}")
                print(f"  BACC:  mean={bacc_mean:.4f}")
                ckpt_results[setting] = {
                    "auroc_mean": mean,
                    "auroc_median": median,
                    "bacc_mean": bacc_mean,
                    "n_systems": int(n_sys),
                    "n_homo": int(r.get("auroc/test_n_homo", 0)),
                    "n_hetero": int(r.get("auroc/test_n_hetero", 0)),
                    "auroc_homo_mean": r.get("auroc/test_homo_mean", float("nan")),
                    "auroc_hetero_mean": r.get("auroc/test_hetero_mean", float("nan")),
                }

        # Summary table
        header = f"{'':^8} | {'AUROC mean':^10} | {'AUROC med':^10} | {'BACC mean':^10} | {'Homo':^10} | {'Hetero':^10} | {'N':^7}"
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"SUMMARY: {ckpt_label}")
        print(sep)
        print(header)
        print(sep)
        for setting in ["holo", "apo", "af2"]:
            r = ckpt_results.get(setting)
            if r is None:
                print(
                    f"{setting:^8} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'N/A':^10} | {'0':^7}"
                )
            else:
                homo = f"{r['auroc_homo_mean']:.4f}" if r["n_homo"] > 0 else "N/A"
                hetero = f"{r['auroc_hetero_mean']:.4f}" if r["n_hetero"] > 0 else "N/A"
                print(
                    f"{setting:^8} | {r['auroc_mean']:.4f}    | {r['auroc_median']:.4f}    | {r['bacc_mean']:.4f}    | {homo:^10} | {hetero:^10} | {r['n_homo']}/{r['n_hetero']}"
                )
        print(sep)


if __name__ == "__main__":
    # Handle --resume flag manually to avoid Hydra conflict
    if "--resume" in sys.argv:
        sys.argv.remove("--resume")
        os.environ["ATOMSURF_RESUME"] = "True"
    main()
