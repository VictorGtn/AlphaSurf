"""
PyTorch Lightning module for MaSIF-Ligand training.
"""

import os
import time

import torch

from alphasurf.tasks.masif_ligand_new.model import MasifLigandNet
from alphasurf.utils.learning_utils import AtomPLModule
from alphasurf.utils.metrics import multi_class_eval


class MasifLigandModule(AtomPLModule):
    """
    Lightning module for MaSIF-Ligand task.

    Handles training/validation/test steps with multi-class classification metrics.
    """

    def __init__(self, cfg):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.model = MasifLigandNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)

    def step(self, batch):
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None

        labels = batch.label
        outputs = self(batch)
        loss = self.criterion(outputs, labels)
        return loss, outputs, labels

    def get_metrics(self, logits, labels, prefix):
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        (
            _,  # accuracy_macro
            _,  # accuracy_micro
            accuracy_balanced,
            _,  # precision_macro
            precision_micro,
            _,  # recall_macro
            recall_micro,
            _,  # f1_macro
            f1_micro,
            auroc_macro,
        ) = multi_class_eval(logits, labels, K=7)

        self.log_dict(
            {
                f"accuracy_balanced/{prefix}": accuracy_balanced,
                f"precision_micro/{prefix}": precision_micro,
                f"recall_micro/{prefix}": recall_micro,
                f"f1_micro/{prefix}": f1_micro,
                f"auroc_macro/{prefix}": auroc_macro,
            },
            on_epoch=True,
            batch_size=len(logits),
        )

    @staticmethod
    def _timing_enabled():
        return os.environ.get("TIMING", "0") == "1"

    def on_train_batch_start(self, batch, batch_idx):
        if not self._timing_enabled():
            return
        if not hasattr(self, "_timing_reset"):
            from alphasurf.utils.timing_stats import reset

            reset()
            self._timing_reset = True
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._train_t0 = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self._timing_enabled() or batch is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._train_t0
        n_proteins = batch.num_graphs

        from alphasurf.utils.timing_stats import print_summary, record

        record("train_fwd_bwd", elapsed)
        record("batch_size", n_proteins)
        record("train_per_protein", elapsed / n_proteins)
        if (batch_idx + 1) % 100 == 0:
            print(f"\n[Timing] batch {batch_idx + 1} (proteins={n_proteins}):")
            print_summary()

    def on_train_end(self):
        if self._timing_enabled():
            from alphasurf.utils.timing_stats import print_summary

            print_summary()
