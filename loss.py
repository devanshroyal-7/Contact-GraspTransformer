import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cgn_heads import (
    NUM_WIDTH_BINS, GRIPPER_WIDTH_MAX,
    PANDA_FINGER_BASE, PANDA_FINGER_TIP, PANDA_BASELINE_DIST,
)


def _width_to_bin_labels(width: torch.Tensor,
                         num_bins: int = NUM_WIDTH_BINS,
                         wmax: float = GRIPPER_WIDTH_MAX) -> torch.Tensor:
    """Convert continuous width values to one-hot bin labels."""
    bin_width = wmax / num_bins
    idx = (width / bin_width).long().clamp(0, num_bins - 1)
    return F.one_hot(idx, num_classes=num_bins).float()


# 5 canonical gripper keypoints in the gripper base frame (paper Fig. 3).
# Columns: wrist, left-finger-base, right-finger-base, left-fingertip, right-fingertip.
# These are parameterised by (approach=a, baseline=b, width=w, contact=c).
# After transformation they become v_i in R^{5x3}.

def _gripper_keypoints(contact, approach, baseline, width):
    """Compute 5 gripper keypoints from grasp parameters (paper Eq. 1-2, Fig. 3).

    All inputs have shape (K, 3) except *width* which is (K,).
    Returns (K, 5, 3).
    """
    half_w = (width / 2).unsqueeze(-1)  # (K, 1)
    d = PANDA_BASELINE_DIST

    wrist = contact + half_w * baseline + d * approach       # (K, 3)
    fb_l = wrist + PANDA_FINGER_BASE * approach + half_w * baseline
    fb_r = wrist + PANDA_FINGER_BASE * approach - half_w * baseline
    ft_l = wrist + PANDA_FINGER_TIP * approach + half_w * baseline
    ft_r = wrist + PANDA_FINGER_TIP * approach - half_w * baseline

    return torch.stack([wrist, fb_l, fb_r, ft_l, ft_r], dim=1)  # (K, 5, 3)


class CGNLoss(nn.Module):
    """Paper-faithful Contact-GraspNet loss (Section III.D).

    ``l = alpha * l_bce,k  +  beta * l_add-s  +  gamma * l_width``

    * **l_bce,k**: top-k (k=512) hard-example-mined BCE for contact confidence.
    * **l_add-s**: confidence-weighted, symmetry-aware average distance between
      5 predicted and GT gripper keypoints (Eq. 8).
    * **l_width**: weighted multi-label BCE over 10 equidistant width bins.
    """

    def __init__(self, adds_weight=10.0, width_weight=1.0, topk=512,
                 num_width_bins=NUM_WIDTH_BINS,
                 gripper_width_max=GRIPPER_WIDTH_MAX):
        super().__init__()
        self.adds_weight = adds_weight
        self.width_weight = width_weight
        self.topk = topk
        self.num_width_bins = num_width_bins
        self.gripper_width_max = gripper_width_max

    # ------------------------------------------------------------------
    def forward(self, preds, targets):
        """
        preds:   dict from ContactGraspNet.forward()
        targets: dict from CGNDataset.__getitem__()  (must include 'points')
        """
        dev = preds['confidence_logits'].device

        # ---------- 1. Top-k confidence BCE (paper: k=512) ----------
        loss_conf = self._topk_bce(preds['confidence_logits'],
                                   targets['confidence'])

        # ---------- Mask: only positive contact points for geometry ----------
        mask = targets['confidence'] > 0.5
        loss_adds  = torch.tensor(0.0, device=dev)
        loss_width = torch.tensor(0.0, device=dev)

        if mask.any():
            # ---------- 2. ADD-S loss (paper Eq. 7-8) ----------
            loss_adds = self._adds_loss(
                contact=targets['points'][mask],
                pred_app=preds['approach_dirs'][mask],
                pred_base=preds['base_dirs'][mask],
                pred_width=preds['widths'][mask],
                pred_conf=preds['confidence'][mask],
                targ_app=targets['approach_dirs'][mask],
                targ_base=targets['base_dirs'][mask],
                targ_width=targets['widths'][mask],
            )

            # ---------- 3. Binned width loss ----------
            width_logits = preds['width_bin_logits'][mask]
            targ_width = targets['widths'][mask]
            bin_labels = _width_to_bin_labels(
                targ_width, self.num_width_bins, self.gripper_width_max)

            bin_counts = bin_labels.sum(dim=0) + 1.0
            bin_weights = bin_counts.sum() / (self.num_width_bins * bin_counts)

            loss_width = F.binary_cross_entropy_with_logits(
                width_logits, bin_labels,
                weight=bin_weights.unsqueeze(0).expand_as(width_logits),
            )

        total = (loss_conf
                 + self.adds_weight * loss_adds
                 + self.width_weight * loss_width)

        return {
            'loss': total,
            'l_conf': loss_conf,
            'l_adds': loss_adds,
            'l_width': loss_width,
        }

    # ------------------------------------------------------------------
    def _topk_bce(self, logits, target):
        """BCE on the k points with the largest per-point error."""
        per_point = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none')          # (B, N)
        B, N = per_point.shape
        k = min(self.topk, N)
        topk_vals, _ = per_point.reshape(B, -1).topk(k, dim=1)
        return topk_vals.mean()

    # ------------------------------------------------------------------
    @staticmethod
    def _adds_loss(contact, pred_app, pred_base, pred_width,
                   pred_conf, targ_app, targ_base, targ_width):
        """Confidence-weighted average gripper-keypoint distance (Eq. 8).

        With the current per-point single-GT assignment, min_u collapses to
        the assigned GT grasp so no explicit minimum search is needed.
        """
        v_pred = _gripper_keypoints(contact, pred_app, pred_base, pred_width)
        v_gt = _gripper_keypoints(contact, targ_app, targ_base, targ_width)

        per_point_dist = (v_pred - v_gt).norm(dim=-1).mean(dim=-1)  # (K,)
        weighted = pred_conf.detach() * per_point_dist
        return weighted.mean()
