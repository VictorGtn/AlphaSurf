"""
PyTorch Lightning module for PINDER-Pair training.

Uses epoch-level AUROC: accumulates all logits/labels across the epoch,
then computes a single global AUROC. This avoids per-batch averaging which
significantly inflates the metric with small/variable batch sizes.
"""

import logging

import torch
from alphasurf.tasks.pinder_pair.loss import PinderPairLoss
from alphasurf.tasks.pinder_pair.model import PinderPairNet, _offset_and_concat
from alphasurf.utils.learning_utils import AtomPLModule
from alphasurf.utils.metrics import compute_accuracy, compute_auroc

# from torch_geometric.nn import global_mean_pool # Removed as protein-level pooling is no longer used

logger = logging.getLogger(__name__)


class PinderPairModule(AtomPLModule):
    """
    Lightning module for PINDER protein-protein interaction task.

    Binary classification: does this residue pair form an interface contact?
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        # New Custom Loss (Simplified)
        self.criterion = PinderPairLoss(
            lambda_site=getattr(cfg, "lambda_site", 4.0),
            lambda_norm=getattr(cfg, "lambda_norm", 1.0),
            gamma=getattr(cfg, "gamma", 2.0),
        )

        self.model = PinderPairNet(cfg_encoder=cfg.encoder, cfg_head=cfg.cfg_head)

        # Accumulators for epoch-level AUROC (avoid per-batch averaging)
        self._val_logits = []
        self._val_labels = []
        self._test_logits = []
        self._test_labels = []

    def _get_site_labels(
        self, batch, num_nodes_1, num_nodes_2, idx_left, idx_right, pair_labels
    ):
        """
        Derive binding site labels (1 if in any positive pair) from pair labels.
        """
        # Initialize with zeros
        site_labels_1 = torch.zeros(num_nodes_1, device=pair_labels.device)
        site_labels_2 = torch.zeros(num_nodes_2, device=pair_labels.device)

        # Identify positive pairs
        pos_mask = pair_labels == 1

        if pos_mask.sum() > 0:
            pos_idx_l = idx_left[pos_mask]
            pos_idx_r = idx_right[pos_mask]

            # Scatter ones (handling duplicates)
            site_labels_1[pos_idx_l] = 1.0
            site_labels_2[pos_idx_r] = 1.0

        return site_labels_1, site_labels_2

    def step(self, batch):
        if batch is None or batch.num_graphs < self.hparams.cfg.min_batch_size:
            return None, None, None, {}

        # Get pair labels and flatten
        if isinstance(batch.label, list):
            labels = torch.cat(batch.label)
        else:
            labels = batch.label
        labels = labels.view(-1)

        # Forward pass
        out = self.model(batch)

        # --- Graph Loss ---
        graph_out = out["graph"]
        emb_left = graph_out["emb_left"]
        emb_right = graph_out["emb_right"]
        site_pred_1 = graph_out["site_pred_1"]
        site_pred_2 = graph_out["site_pred_2"]

        # Reconstruct indices for site labels
        base_left = torch.cumsum(batch.g1_len, dim=0)
        base_right = torch.cumsum(batch.g2_len, dim=0)
        idx_left = _offset_and_concat(batch.idx_left, base_left, batch.label)
        idx_right = _offset_and_concat(batch.idx_right, base_right, batch.label)

        # Compute binding site labels (ground truth)
        site_labels_1, site_labels_2 = self._get_site_labels(
            batch, batch.g1_len.sum(), batch.g2_len.sum(), idx_left, idx_right, labels
        )

        # Calculate Graph Loss
        loss_graph, loss_dict_graph = self.criterion(
            site_pred_left=site_pred_1,
            site_pred_right=site_pred_2,
            site_labels_left=site_labels_1,
            site_labels_right=site_labels_2,
            emb_left=emb_left,
            emb_right=emb_right,
            pair_labels=labels,
        )

        # Calculate Surface Loss (if available)
        loss_surf = 0.0
        surf_out = out["surface"]
        if surf_out is not None:
            # Reconstruct indices for surface
            s1_lens = torch.tensor(
                [s.x.shape[0] for s in batch.surface_1.to_data_list()],
                device=batch.surface_1.x.device,
            )
            s2_lens = torch.tensor(
                [s.x.shape[0] for s in batch.surface_2.to_data_list()],
                device=batch.surface_2.x.device,
            )
            base_l_surf = torch.cumsum(s1_lens, dim=0)
            base_r_surf = torch.cumsum(s2_lens, dim=0)

            surf_idx_l = _offset_and_concat(
                batch.surface_idx_left, base_l_surf, batch.surface_label
            )
            surf_idx_r = _offset_and_concat(
                batch.surface_idx_right, base_r_surf, batch.surface_label
            )

            if hasattr(batch, "surface_label"):
                s_labels = batch.surface_label
                if isinstance(s_labels, list):
                    s_labels = torch.cat(s_labels)
                s_labels = s_labels.view(-1).to(emb_left.device)

                surf_site_labels_1, surf_site_labels_2 = self._get_site_labels(
                    batch,
                    surf_out["site_pred_1"].shape[0],
                    surf_out["site_pred_2"].shape[0],
                    surf_idx_l,
                    surf_idx_r,
                    s_labels,
                )

                # Check / Init separate surface criterion
                if not hasattr(self, "surface_criterion"):
                    self.surface_criterion = PinderPairLoss(
                        lambda_site=self.criterion.lambda_site,
                        lambda_norm=self.criterion.lambda_norm,
                        gamma=self.criterion.focal.gamma,
                    ).to(self.device)

                loss_s, loss_dict_s = self.surface_criterion(
                    site_pred_left=surf_out["site_pred_1"],
                    site_pred_right=surf_out["site_pred_2"],
                    site_labels_left=surf_site_labels_1,
                    site_labels_right=surf_site_labels_2,
                    emb_left=surf_out["emb_left"],
                    emb_right=surf_out["emb_right"],
                    pair_labels=s_labels,
                )
                loss_surf = loss_s

                # Merge dicts
                for k, v in loss_dict_s.items():
                    loss_dict_graph[f"{k}_surf"] = v

        # Logits for metrics: distinct from loss
        # Use simple dot product as similarity score
        logits = (emb_left * emb_right).sum(dim=1)

        # Total Loss
        surface_loss_weight = getattr(self.hparams.cfg, "surface_loss_weight", 1.0)
        total_loss = loss_graph + surface_loss_weight * loss_surf

        loss_dict_graph["loss_graph"] = loss_graph
        loss_dict_graph["loss_surf"] = loss_surf

        return total_loss, logits, labels, loss_dict_graph

    def training_step(self, batch, batch_idx):
        # We need a surface criterion
        if not hasattr(self, "surface_criterion"):
            self.surface_criterion = PinderPairLoss(
                lambda_site=self.criterion.lambda_site,
                lambda_norm=self.criterion.lambda_norm,
                gamma=self.criterion.focal.gamma,
            ).to(self.device)

        loss, logits, labels, loss_dict = self.step(batch)
        if loss is None:
            return None

        self.log(
            "loss/train",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(logits),
        )

        # Log components
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                self.log(f"loss/{k}", v.item(), on_step=True, batch_size=len(logits))

        # Metrics
        acc = compute_accuracy(logits, labels, add_sigmoid=True)
        auroc = compute_auroc(logits, labels)

        self.log_dict(
            {"acc/train": acc, "auroc/train": auroc},
            on_epoch=True,
            batch_size=len(logits),
        )

        return loss

    # ── Validation ──────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_logits = []
        self._val_labels = []
        # Support for Test-during-Val
        self._test_val_logits = []
        self._test_val_labels = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Lazy init surface criterion for val too
        if not hasattr(self, "surface_criterion"):
            self.surface_criterion = PinderPairLoss(
                lambda_site=self.criterion.lambda_site,
                lambda_norm=self.criterion.lambda_norm,
                gamma=self.criterion.focal.gamma,
            ).to(self.device)

        self.model.eval()
        loss, logits, labels, loss_dict = self.step(batch)

        if loss is None:
            return None

        # Determine which set we are processing
        is_val = dataloader_idx == 0
        prefix = "val" if is_val else "test_val"

        self.log(
            f"loss/{prefix}",
            loss.item(),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(logits),
            add_dataloader_idx=False,
        )

        # Accumulate for global AUROC
        if is_val:
            self._val_logits.append(logits.detach().cpu())
            self._val_labels.append(labels.detach().cpu())
        else:
            self._test_val_logits.append(logits.detach().cpu())
            self._test_val_labels.append(labels.detach().cpu())

    def on_validation_epoch_end(self):
        # 1. Process Validation Set (Index 0)
        if self._val_logits:
            all_logits = torch.cat(self._val_logits)
            all_labels = torch.cat(self._val_labels)

            auroc = compute_auroc(all_logits, all_labels)
            acc = compute_accuracy(all_logits, all_labels, add_sigmoid=True)

            self.log("auroc/val", auroc, prog_bar=True)
            self.log("acc/val", acc, prog_bar=True)
            self.log(
                "auroc_val", auroc, prog_bar=True, logger=False
            )  # Checkpoint monitor

            self._val_logits.clear()
            self._val_labels.clear()
        else:
            # Fallback if empty
            self.log("auroc/val", 0.5, prog_bar=True)
            self.log("auroc_val", 0.5, prog_bar=True, logger=False)

        # 2. Process Test Set (Index 1) - if present
        if self._test_val_logits:
            all_logits_test = torch.cat(self._test_val_logits)
            all_labels_test = torch.cat(self._test_val_labels)

            auroc_test = compute_auroc(all_logits_test, all_labels_test)
            acc_test = compute_accuracy(
                all_logits_test, all_labels_test, add_sigmoid=True
            )

            self.log("auroc/test_val", auroc_test, prog_bar=True)
            self.log("acc/test_val", acc_test, prog_bar=True)

            self._test_val_logits.clear()
            self._test_val_labels.clear()

    # ── Test ────────────────────────────────────────────────────

    def on_test_epoch_start(self):
        self._test_logits.clear()
        self._test_labels.clear()

    def test_step(self, batch, batch_idx):
        # Lazy init surface criterion for test too
        if not hasattr(self, "surface_criterion"):
            self.surface_criterion = PinderPairLoss(
                lambda_site=self.criterion.lambda_site,
                lambda_norm=self.criterion.lambda_norm,
                gamma=self.criterion.focal.gamma,
            ).to(self.device)

        self.model.eval()
        loss, logits, labels, loss_dict = self.step(batch)

        if loss is None:
            return None

        self.log(
            "loss/test",
            loss.item(),
            on_epoch=True,
            prog_bar=True,
            batch_size=len(logits),
        )

        # Accumulate for global AUROC
        self._test_logits.append(logits.detach().cpu())
        self._test_labels.append(labels.detach().cpu())

    def on_test_epoch_end(self):
        if not self._test_logits:
            self.log("auroc/test", 0.5, prog_bar=True)
            return

        all_logits = torch.cat(self._test_logits)
        all_labels = torch.cat(self._test_labels)

        auroc = compute_auroc(all_logits, all_labels)

        self.log("auroc/test", auroc, prog_bar=True)

        self._test_logits.clear()
        self._test_labels.clear()
