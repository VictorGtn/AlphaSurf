import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    L(p, y) = -y(1-p)^gamma log(p) - (1-y)p^gamma log(1-p)
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (N, *) raw scores (before sigmoid)
            targets: (N, *) binary labels (0 or 1)
        """
        if targets.shape != logits.shape:
            targets = targets.view_as(logits)
        # Ensure targets are float
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        return loss.mean()


class PinderPairLoss(nn.Module):
    """
    Combined loss for Pinder Pair task:
    L_total = L_site + L_comp (Focal only) + L_norm
    """

    def __init__(self, lambda_site=4.0, lambda_norm=1.0, gamma=2.0):
        super().__init__()
        self.lambda_site = lambda_site
        self.lambda_norm = lambda_norm

        self.focal = FocalLoss(gamma=gamma)

    def forward(
        self,
        site_pred_left,
        site_pred_right,
        site_labels_left,
        site_labels_right,
        emb_left,
        emb_right,
        pair_labels,
    ):
        # 1. Binding Site Loss (L_site)
        l_site_left = self.focal(site_pred_left, site_labels_left)
        l_site_right = self.focal(site_pred_right, site_labels_right)
        l_site = self.lambda_site * (l_site_left + l_site_right)

        # 2. Pairwise Complementarity Loss (L_comp)
        # Focal on dot product (treating dot prod as logit)
        dot_sim = (emb_left * emb_right).sum(dim=1, keepdim=True)
        # Squeeze last dim for focal loss input if needed, but BCEWithLogits accepts (N, 1)
        l_comp = self.focal(dot_sim, pair_labels.reshape(-1, 1).float())

        # 3. Unit Norm Regularization
        # Penalize deviation from unit norm
        norm_l = torch.norm(emb_left, p=2, dim=1)
        norm_r = torch.norm(emb_right, p=2, dim=1)
        l_norm = ((norm_l - 1.0) ** 2).mean() + ((norm_r - 1.0) ** 2).mean()

        total_loss = l_site + l_comp + self.lambda_norm * l_norm

        return total_loss, {
            "l_site": l_site,
            "l_comp": l_comp,
            "l_norm": l_norm,
        }
