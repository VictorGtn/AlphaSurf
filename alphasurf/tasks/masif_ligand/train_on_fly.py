# std
import os
import sys
from pathlib import Path

# Set environment variable to suppress warnings in all processes (including multiprocessing workers)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import warnings

# Silence noisy warnings
warnings.filterwarnings("ignore")

# 3p
import hydra
import torch
from omegaconf import OmegaConf

# Fix PyTorch 2.6+ checkpoint loading (allow all types for trusted checkpoints)
# PyTorch 2.6+ defaults to weights_only=True, but Lightning checkpoints contain omegaconf configs
# Since these are our own checkpoints, we can safely disable weights_only restriction
_original_load = torch.load


def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_false


import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# project
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.network_utils.misc_arch.timing import (
    create_timing_callback,
    enable_timing,
)
from alphasurf.utils.callbacks import CommandLoggerCallback, add_wandb_logger
from alphasurf.utils.timing_stats import TimingStats
from data_loader_on_fly import MasifLigandDataModuleOnFly  # Use on-fly data loader
from pl_model import MasifLigandModule

# torch.multiprocessing.set_sharing_strategy('file_system')
# Limit PyTorch threads to prevent contention between DataLoader workers
# This is critical for on-the-fly preprocessing - without it, 8 workers each spawn
# multiple threads causing massive slowdown
torch.set_num_threads(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    """
    Training script with on-the-fly data preprocessing.

    This script generates surfaces, graphs, and ESM embeddings during training,
    eliminating the need for a separate preprocessing step.

    Configure on-fly preprocessing via the `on_fly` config section:
        on_fly.surface_method: 'msms' or 'alpha_complex'
        on_fly.alpha_value: Alpha value for alpha complex
        on_fly.face_reduction_rate: Mesh simplification rate

    """
    # Enable timing collection for performance monitoring
    enable_timing(category=cfg.timing)
    # Clear timing stats from previous runs
    TimingStats.reset()

    command = f"python3 {' '.join(sys.argv)}"
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)

    seed = cfg.seed
    pl.seed_everything(seed, workers=True)

    device = (
        torch.device("cuda", cfg.device)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print("\n" + "=" * 60)
    print("ON-THE-FLY TRAINING MODE")
    print("=" * 60)
    print("Surfaces and graphs will be generated during training.")
    if hasattr(cfg, "on_fly"):
        print(f"Surface method: {getattr(cfg.on_fly, 'surface_method', 'msms')}")
        print(f"Face reduction rate: {getattr(cfg.on_fly, 'face_reduction_rate', 0.1)}")

    print("=" * 60 + "\n")

    # init datamodule (using on-fly version)
    datamodule = MasifLigandDataModuleOnFly(cfg)

    # init model
    model = MasifLigandModule(cfg)

    version = TensorBoardLogger(save_dir=cfg.log_dir).version
    version_name = (
        f"version_{version}_{cfg.run_name}"
        if not cfg.use_wandb
        else f"version_{version}_{cfg.run_name}"
    )
    tb_logger = TensorBoardLogger(save_dir=cfg.log_dir, version=version_name)
    loggers = [tb_logger]

    if cfg.use_wandb:
        # Include data directory in wandb run name
        data_dir_name = Path(cfg.data_dir).name if cfg.data_dir else "default"
        wandb_run_name = f"{cfg.run_name}_{data_dir_name}_on_fly"
        project_name = getattr(cfg, "project_name", "masif_ligand")
        add_wandb_logger(loggers, projectname=project_name, runname=wandb_run_name)

    # callbacks
    lr_logger = pl.callbacks.LearningRateMonitor()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="{epoch}-{accuracy_balanced/val:.2f}",
        dirpath=Path(tb_logger.log_dir) / "checkpoints",
        monitor=cfg.train.to_monitor,
        mode="max",
        save_last=True,
        save_top_k=cfg.train.save_top_k,
        verbose=False,
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=cfg.train.to_monitor,
        patience=cfg.train.early_stoping_patience,
        mode="max",
    )

    # Timing statistics callback
    timing_callback = create_timing_callback()

    callbacks = [
        lr_logger,
        checkpoint_callback,
        early_stop_callback,
        CommandLoggerCallback(command),
        timing_callback,
    ]

    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [cfg.device]}
    else:
        params = {}

    # Setup profiler if requested
    profiler = None
    if getattr(cfg.train, "profile", False):
        from pytorch_lightning.profilers import PyTorchProfiler

        profiler = PyTorchProfiler(
            dirpath=Path(tb_logger.log_dir) / "profiler",
            filename="profile",
            schedule=torch.profiler.schedule(wait=1, warmup=2, active=5, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(Path(tb_logger.log_dir) / "profiler_tb")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        print(f"[PROFILER] Enabled - output to {Path(tb_logger.log_dir) / 'profiler'}")

    # init trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        profiler=profiler,
        # epochs, batch size and when to val
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        val_check_interval=cfg.train.val_check_interval,
        # just verbose to maybe be used
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        # auto_lr_find=cfg.train.auto_lr_find,
        log_every_n_steps=cfg.train.log_every_n_steps,
        max_steps=cfg.train.max_steps,
        # gradient clipping
        gradient_clip_val=cfg.train.gradient_clip_val,
        # detect NaNs
        detect_anomaly=cfg.train.detect_anomaly,
        # debugging
        overfit_batches=cfg.train.overfit_batches,
        # gpu
        **params,
    )

    # train
    ckpt_path = getattr(cfg, "ckpt_path", None)
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # test
    print("*****************test best ckpt*****************")
    results = trainer.test(model, ckpt_path="best", datamodule=datamodule)
    acc_balanced_best = results[0]["accuracy_balanced/test"]
    trainer.logger.log_metrics({"accuracy_balanced/test_best": acc_balanced_best})
    print("*****************test last ckpt*****************")
    results = trainer.test(model, ckpt_path="last", datamodule=datamodule)
    acc_balanced_last = results[0]["accuracy_balanced/test"]
    trainer.logger.log_metrics({"accuracy_balanced/test_last": acc_balanced_last})


if __name__ == "__main__":
    main()
