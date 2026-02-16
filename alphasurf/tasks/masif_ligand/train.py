# std
import sys
import os
from pathlib import Path

# Set environment variable to suppress warnings in all processes (including multiprocessing workers)
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Ensure pykeops cache directory exists before any pykeops imports
# Pykeops creates system-specific subdirectories that must exist before it can create temp files
import platform

cache_base = os.path.expanduser("~/.cache/keops2.3")
os.makedirs(cache_base, exist_ok=True)

# Pykeops creates subdirectories based on system info. Pre-create likely directory structure
# Format appears to be: OS_hostname_kernel_python_version (CUDA_VISIBLE_DEVICES may be appended)
# Based on error, it's: Linux_node006_5.4.0-187-generic_p3.10.19
python_version = f"p{platform.python_version()}"
hostname = platform.node()
kernel = platform.release()
os_name = platform.system()

# Create the system-specific directory that pykeops will use (without CUDA_VISIBLE_DEVICES first)
pykeops_dir_base = f"{os_name}_{hostname}_{kernel}_{python_version}"
pykeops_cache_dir = os.path.join(cache_base, pykeops_dir_base)
os.makedirs(pykeops_cache_dir, exist_ok=True)

# Also create with CUDA_VISIBLE_DEVICES suffix if it exists
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if cuda_visible:
    pykeops_dir_with_cuda = f"{pykeops_dir_base}_CUDA_VISIBLE_DEVICES_{cuda_visible}"
    pykeops_cache_dir_cuda = os.path.join(cache_base, pykeops_dir_with_cuda)
    os.makedirs(pykeops_cache_dir_cuda, exist_ok=True)

import warnings

# Silence noisy warnings
warnings.filterwarnings("ignore")

# 3p
import hydra
from omegaconf import OmegaConf
import torch

# Set sharing strategy to file_system to avoid shared memory limits (Bus Error)
# This must be done before any large tensors are passed between processes
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass


# Fix PyTorch 2.6+ checkpoint loading (allow all types for trusted checkpoints)
# PyTorch 2.6+ defaults to weights_only=True, but Lightning checkpoints contain omegaconf configs
# Since these are our own checkpoints, we can safely disable weights_only restriction
_original_load = torch.load


def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_false


def _warmup_keops(device):
    try:
        from pykeops.torch import LazyTensor
    except ImportError:
        return
    x = torch.randn(128, 3, device=device)
    y = torch.randn(128, 3, device=device)
    x_i = LazyTensor(x[:, None, :])
    y_j = LazyTensor(y[None, :, :])
    diff = x_i - y_j
    k = (-(diff * diff).sum(-1)).exp()
    _ = k.sum(dim=1)
    if device.type == "cuda":
        torch.cuda.synchronize()


import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

# project
if __name__ == "__main__":
    sys.path.append(str(Path(__file__).absolute().parents[3]))

from alphasurf.utils.callbacks import (
    CommandLoggerCallback,
    add_wandb_logger,
)
from alphasurf.network_utils.misc_arch.timing import (
    enable_timing,
    create_timing_callback,
)
from pl_model import MasifLigandModule
from data_loader import MasifLigandDataModule


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg=None):
    # Enable timing collection for performance monitoring
    enable_timing(category=cfg.timing)

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
    _warmup_keops(device)

    # init datamodule
    datamodule = MasifLigandDataModule(cfg)

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
        data_dir_name = Path(cfg.data_dir).name
        wandb_run_name = f"{cfg.run_name}_{data_dir_name}"
        add_wandb_logger(loggers, projectname="masif_ligand", runname=wandb_run_name)

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
    ]  # , GPUMemoryCallback()]

    if torch.cuda.is_available():
        params = {"accelerator": "gpu", "devices": [cfg.device]}
    else:
        params = {}

    # init trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
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
