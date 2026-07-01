"""
PyTorch Lightning module for PINDER-Pair training.

Per-system AUROC: accumulates logits/labels per system across the epoch,
then computes AUROC per system and reports mean/median/std.
"""

import logging
import os
import time

import numpy as np
import torch
from alphasurf.tasks.pinder_pair.loss import FocalLoss, PinderPairLoss
from alphasurf.tasks.pinder_pair.model import PinderPairNet, _offset_and_concat
from alphasurf.utils.learning_utils import AtomPLModule
from alphasurf.utils.metrics import compute_accuracy, compute_auroc
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


class PinderPairModule(AtomPLModule):
    """
    Lightning module for PINDER protein-protein interaction task.

    Binary classification: does this residue pair form an interface contact?
    """

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()

        self.loss_mode = getattr(cfg, "loss_mode", "metric")

        if self.loss_mode == "focal":
            self.criterion = FocalLoss(gamma=getattr(cfg, "gamma", 2.0))
        else:
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

        if isinstance(batch.label, list):
            labels = torch.cat(batch.label)
        else:
            labels = batch.label
        labels = labels.view(-1)

        if self.loss_mode == "focal":
            return self._step_focal(batch, labels)
        else:
            return self._step_metric(batch, labels)

    def _step_focal(self, batch, labels):
        """Focal mode: pair classification with focal loss on MLP heads."""
        out = self.model(batch)
        graph_out = out["graph"]

        # Graph pair loss
        pair_logits = graph_out["pair_logit"].squeeze(-1)
        loss_graph = self.criterion(pair_logits, labels.float())

        # Surface pair loss
        loss_surf = torch.tensor(0.0, device=labels.device)
        surf_out = out["surface"]
        if surf_out is not None and hasattr(batch, "surface_label"):
            s_labels = batch.surface_label
            if isinstance(s_labels, list):
                s_labels = torch.cat(s_labels)
            s_labels = s_labels.view(-1).to(pair_logits.device)
            surf_pair_logits = surf_out["pair_logit"].squeeze(-1)
            loss_surf = self.criterion(surf_pair_logits, s_labels.float())

        surface_loss_weight = getattr(self.hparams.cfg, "surface_loss_weight", 1.0)
        total_loss = loss_graph + surface_loss_weight * loss_surf

        loss_dict = {
            "loss_pair_graph": loss_graph,
            "loss_pair_surf": loss_surf,
        }

        return total_loss, pair_logits, labels, loss_dict

    def _step_metric(self, batch, labels):
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

    def _timing_enabled(self):
        return os.environ.get("TIMING", "0") == "1"

    def on_train_batch_start(self, batch, batch_idx):
        if self._timing_enabled():
            if not hasattr(self, "_timing_reset"):
                from alphasurf.utils.timing_stats import reset

                reset()
                self._timing_reset = True
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._train_t0 = time.perf_counter()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self._timing_enabled():
            return
        if batch is None:
            return
        if hasattr(self, "_train_t0"):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - self._train_t0
            from alphasurf.utils.timing_stats import record

            record("train_fwd_bwd", elapsed)
            n_systems = batch["graph_1"].num_graphs
            n_proteins = n_systems * 2
            per_protein = elapsed / n_proteins
            record("batch_size", n_systems)
            record("train_per_protein", per_protein)
            self.log(
                "timing/train_fwd_bwd",
                elapsed,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=1,
            )
            self.log(
                "timing/train_per_protein",
                per_protein,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                batch_size=1,
            )
            if (batch_idx + 1) % 100 == 0:
                from alphasurf.utils.timing_stats import print_summary

                print(f"\n[Timing] batch {batch_idx + 1} (systems={n_systems}):")
                print_summary()

    def training_step(self, batch, batch_idx):
        # Lazy init surface criterion (metric mode only)
        if self.loss_mode == "metric" and not hasattr(self, "surface_criterion"):
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

    def on_train_end(self):
        from alphasurf.utils.timing_stats import print_summary

        print_summary()

    # ── Validation ──────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self._val_logits = []
        self._val_labels = []
        # Support for Test-during-Val
        self._test_val_logits = []
        self._test_val_labels = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Lazy init surface criterion (metric mode only)
        if self.loss_mode == "metric" and not hasattr(self, "surface_criterion"):
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
        self._test_per_system = []

    def test_step(self, batch, batch_idx):
        # Lazy init surface criterion (metric mode only)
        if self.loss_mode == "metric" and not hasattr(self, "surface_criterion"):
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

        # Accumulate per-system for per-system AUROC
        if isinstance(batch.label, list):
            per_system_labels = [lab.reshape(-1) for lab in batch.label]
            sys_ids = (
                batch.system_id
                if hasattr(batch, "system_id")
                else [None] * len(per_system_labels)
            )
            if isinstance(sys_ids, str):
                sys_ids = [sys_ids]

            offset = 0
            for si, sys_labels in enumerate(per_system_labels):
                n_pairs = len(sys_labels)
                sys_logits = logits[offset : offset + n_pairs].detach().cpu()
                sys_lab = sys_labels.detach().cpu()
                sid = sys_ids[si] if si < len(sys_ids) else f"unknown_{batch_idx}_{si}"
                self._test_per_system.append((sid, sys_logits, sys_lab))
                offset += n_pairs

    def on_test_epoch_end(self):
        # Per-system AUROC + balanced accuracy + hetero/homo stratification
        system_aurocs = []
        system_baccs = []
        homo_aurocs = []
        hetero_aurocs = []
        per_system_rows = []

        for sid, sys_logits, sys_labels in self._test_per_system:
            logits_np = sys_logits.numpy().ravel()
            labels_np = sys_labels.numpy().astype(int).ravel()
            n_pos = labels_np.sum()
            n_neg = len(labels_np) - n_pos
            if n_pos > 0 and n_neg > 0:
                auc = roc_auc_score(labels_np, logits_np)
                system_aurocs.append(auc)
                preds = (logits_np > 0).astype(int)
                bacc = balanced_accuracy_score(labels_np, preds)
                system_baccs.append(bacc)
                is_homo = False
                if sid and "--" in sid:
                    parts = sid.split("--")
                    r_uniprot = parts[0].split("_")[-1]
                    l_uniprot = parts[1].split("_")[-1]
                    if r_uniprot == l_uniprot:
                        homo_aurocs.append(auc)
                        is_homo = True
                    else:
                        hetero_aurocs.append(auc)
                per_system_rows.append(
                    {
                        "system_id": sid,
                        "auroc": auc,
                        "bacc": bacc,
                        "is_homodimer": is_homo,
                    }
                )

        if system_aurocs:
            aurocs = np.array(system_aurocs)
            baccs = np.array(system_baccs)
            self.log("auroc/test", aurocs.mean(), prog_bar=True)
            self.log("auroc/test_mean", aurocs.mean())
            self.log("auroc/test_median", float(np.median(aurocs)))
            self.log("auroc/test_std", aurocs.std())
            self.log("auroc/test_n_systems", float(len(aurocs)))
            self.log("bacc/test_mean", baccs.mean())
            self.log("bacc/test_median", float(np.median(baccs)))
            if homo_aurocs:
                self.log("auroc/test_homo_mean", float(np.mean(homo_aurocs)))
                self.log("auroc/test_n_homo", float(len(homo_aurocs)))
            if hetero_aurocs:
                self.log("auroc/test_hetero_mean", float(np.mean(hetero_aurocs)))
                self.log("auroc/test_n_hetero", float(len(hetero_aurocs)))
        elif self._test_logits:
            # Fallback to global AUROC if no per-system data
            all_logits = torch.cat(self._test_logits)
            all_labels = torch.cat(self._test_labels)
            auroc = compute_auroc(all_logits, all_labels)
            self.log("auroc/test", auroc, prog_bar=True)
        else:
            self.log("auroc/test", 0.5, prog_bar=True)

        # Dump per-system results to CSV
        dump_dir = getattr(self.hparams.cfg, "dump_per_system", None)
        if dump_dir and per_system_rows:
            import pandas as pd

            os.makedirs(dump_dir, exist_ok=True)
            test_setting = getattr(self.hparams.cfg, "test_setting", "unknown")
            run_name = getattr(self.hparams.cfg, "run_name", "default")
            csv_path = os.path.join(dump_dir, f"{run_name}_{test_setting}.csv")
            pd.DataFrame(per_system_rows).to_csv(csv_path, index=False)
            print(
                f"  Dumped per-system results: {csv_path} ({len(per_system_rows)} systems)"
            )

        self._test_logits.clear()
        self._test_labels.clear()
        self._test_per_system.clear()
