"""Training and pooled residue-level metrics for MISATO binding sites."""

from __future__ import annotations

import numpy as np
import torch
from alphasurf.tasks.misato_binding_site.model import MisatoBindingSiteNet
from alphasurf.utils.learning_utils import AtomPLModule
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MisatoBindingSiteModule(AtomPLModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = MisatoBindingSiteNet(cfg.encoder, cfg.cfg_head)
        # Matches the paper repository: unweighted two-class cross-entropy.
        self.criterion = torch.nn.CrossEntropyLoss()

    def step(self, batch):
        if batch is None or batch.num_graphs < 1:
            return None, None, None
        logits = self.model(batch)["logits"]
        labels = (
            torch.cat(batch.y).long() if isinstance(batch.y, list) else batch.y.long()
        )
        labels = labels.view(-1)
        if len(labels) != len(logits):
            raise RuntimeError(f"Label/node mismatch: {len(labels)} != {len(logits)}")
        return self.criterion(logits, labels), logits, labels

    def get_metrics(self, logits, labels, prefix):
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0).numpy().astype(np.int64)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = (probs >= 0.5).astype(np.int64)
        metrics = {
            f"accuracy/{prefix}": accuracy_score(labels, preds),
            f"precision/{prefix}": precision_score(labels, preds, zero_division=0),
            f"recall/{prefix}": recall_score(labels, preds, zero_division=0),
            f"f1/{prefix}": f1_score(labels, preds, zero_division=0),
        }
        if np.unique(labels).size == 2:
            metrics[f"auroc/{prefix}"] = roc_auc_score(labels, probs)
            metrics[f"auprc/{prefix}"] = average_precision_score(labels, probs)
        self.log_dict(metrics, on_epoch=True, prog_bar=prefix != "train")
