"""
PyTorch Lightning module for S3F-style pretraining.

Masked residue-type prediction: cross-entropy on the 15% selected positions.
Optimizer: plain Adam (lr 2e-4), no scheduler, no warmup, no weight decay
(per S3F paper §3.5 and config/pretrain/s3f.yaml).
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from alphasurf.tasks.s3f_pretrain.model import S3FPretrainNet
from alphasurf.utils.learning_utils import AtomPLModule

logger = logging.getLogger(__name__)


class S3FPretrainModule(AtomPLModule):
    """Lightning module for S3F-style masked residue prediction."""

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = S3FPretrainNet(cfg.encoder, cfg.cfg_head)
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
        self.log("loss/train", loss, prog_bar=True, batch_size=logits.shape[0])
        self.log("acc/train", extra["acc"], prog_bar=True, batch_size=logits.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        result = self.step(batch)
        if result is None or result[0] is None:
            return None
        loss, logits, targets, extra = result
        self.log("loss/val", loss, prog_bar=True, batch_size=logits.shape[0])
        self.log("acc/val", extra["acc"], prog_bar=True, batch_size=logits.shape[0])

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
