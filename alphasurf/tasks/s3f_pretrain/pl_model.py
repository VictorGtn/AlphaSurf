"""
PyTorch Lightning module for S3F-style pretraining.

Masked residue-type prediction: cross-entropy on the 15% selected positions.
Optimizer: plain Adam (lr 2e-4), no scheduler, no warmup, no weight decay
(per S3F paper §3.5 and config/pretrain/s3f.yaml).
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.nn as nn
from alphasurf.tasks.s3f_pretrain.checkpointing import strip_frozen_esm_weights
from alphasurf.tasks.s3f_pretrain.model import S3FPretrainNet
from alphasurf.utils.learning_utils import AtomPLModule

logger = logging.getLogger(__name__)


class S3FPretrainModule(AtomPLModule):
    """Lightning module for S3F-style masked residue prediction."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = S3FPretrainNet(
            cfg.encoder,
            cfg.cfg_head,
            cfg_surface_esm=getattr(cfg, "surface_esm", None),
        )
        self.criterion = nn.CrossEntropyLoss()

    def step(self, batch):
        if batch is None or batch.num_graphs < 1:
            return None, None, None
        device = self.device
        out = self.model(batch, device)
        logits = out["logits"]
        global_masked = out["global_masked"]
        targets = out["target_residues"]

        if global_masked.numel() == 0:
            return None, logits, targets

        masked_logits = logits[global_masked]
        loss = self.criterion(masked_logits, targets)
        acc = (masked_logits.argmax(dim=-1) == targets).float().mean()
        return loss, logits, targets, {"acc": acc}

    def training_step(self, batch, batch_idx):
        result = self.step(batch)
        if result is None or result[0] is None:
            return None
        loss, logits, targets, extra = result
        metric_batch_size = targets.numel()
        self.log("loss/train", loss, prog_bar=True, batch_size=metric_batch_size)
        self.log(
            "acc/train", extra["acc"], prog_bar=True, batch_size=metric_batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        result = self.step(batch)
        if result is None or result[0] is None:
            return None
        loss, logits, targets, extra = result
        metric_batch_size = targets.numel()
        self.log("loss/val", loss, prog_bar=True, batch_size=metric_batch_size)
        self.log(
            "acc/val", extra["acc"], prog_bar=True, batch_size=metric_batch_size
        )
        # Slash-free alias: ModelCheckpoint interpolates the monitored name into
        # the filename, and a "/" there makes Lightning create a directory per
        # epoch instead of a checkpoint file.
        self.log("acc_val", extra["acc"], batch_size=metric_batch_size)

    def _timing_enabled(self):
        return os.environ.get("TIMING", "0") == "1"

    def on_train_epoch_start(self):
        if not self._timing_enabled():
            return
        if not hasattr(self, "_timing_reset"):
            from alphasurf.utils.timing_stats import reset

            reset()
            self._timing_reset = True
        self._epoch_t0 = time.perf_counter()
        self._epoch_proteins = 0

    def on_train_batch_start(self, batch, batch_idx):
        if not self._timing_enabled():
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._train_t0 = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self._timing_enabled() or batch is None:
            return
        if not hasattr(self, "_train_t0"):
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._train_t0

        from alphasurf.utils.timing_stats import record

        n_proteins = batch.num_graphs
        self._epoch_proteins = getattr(self, "_epoch_proteins", 0) + n_proteins
        record("train_fwd_bwd", elapsed)
        record("batch_size", n_proteins)
        record("train_per_protein", elapsed / n_proteins)
        self.log(
            "timing/train_per_protein",
            elapsed / n_proteins,
            on_step=True,
            on_epoch=False,
            batch_size=1,
        )
        if (batch_idx + 1) % 100 == 0:
            from alphasurf.utils.timing_stats import print_summary

            print(f"\n[Timing] batch {batch_idx + 1} (proteins={n_proteins}):")
            print_summary()

    def on_train_epoch_end(self):
        if not self._timing_enabled() or not hasattr(self, "_epoch_t0"):
            return
        from alphasurf.utils.timing_stats import print_summary, record

        elapsed = time.perf_counter() - self._epoch_t0
        proteins = max(getattr(self, "_epoch_proteins", 0), 1)
        record("train_epoch", elapsed)
        self.log("timing/train_epoch", elapsed, on_epoch=True, batch_size=1)
        print(
            f"\n[Timing] epoch {self.current_epoch}: {elapsed:.1f}s "
            f"({proteins} proteins, {elapsed / proteins * 1000:.1f} ms/protein)"
        )
        print_summary()

    def configure_optimizers(self):
        lr = self.cfg.optimizer.lr
        b1 = getattr(self.cfg.optimizer, "b1", 0.9)
        b2 = getattr(self.cfg.optimizer, "b2", 0.999)
        wd = getattr(self.cfg.optimizer, "weight_decay", 0.0)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd
        )
        return optimizer

    def on_fit_start(self):
        device = self.device
        self.model._load_esm(device)

    def on_save_checkpoint(self, checkpoint):
        removed = strip_frozen_esm_weights(checkpoint)
        if removed:
            logger.info("Excluded %d frozen ESM parameters from checkpoint", removed)

    def on_load_checkpoint(self, checkpoint):
        removed = strip_frozen_esm_weights(checkpoint)
        if removed:
            logger.info("Ignored %d frozen ESM parameters from checkpoint", removed)
