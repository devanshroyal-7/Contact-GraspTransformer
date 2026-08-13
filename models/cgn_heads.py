from __future__ import annotations

from typing import TypeVar, overload

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.types import Predictions

NUM_WIDTH_BINS = 10
GRIPPER_WIDTH_MAX = 0.08

# Panda gripper geometry (metres) -- used by ADD-S loss
PANDA_FINGER_BASE = 0.0584
PANDA_FINGER_TIP = 0.1053
PANDA_BASELINE_DIST = 0.0584  # d: distance from baseline to base frame

ArrayT = TypeVar("ArrayT", torch.Tensor, np.ndarray)


@overload
def wrist_from_grasp(
    contact: torch.Tensor,
    approach: torch.Tensor,
    baseline: torch.Tensor,
    width: torch.Tensor,
) -> torch.Tensor: ...


@overload
def wrist_from_grasp(
    contact: np.ndarray,
    approach: np.ndarray,
    baseline: np.ndarray,
    width: np.ndarray,
) -> np.ndarray: ...


def wrist_from_grasp(
    contact: ArrayT,
    approach: ArrayT,
    baseline: ArrayT,
    width: ArrayT,
) -> ArrayT:
    """Wrist / TCP origin from contact-grasp parameters.

    ``wrist = contact + (width / 2) * baseline + d * approach``
    with ``d = PANDA_BASELINE_DIST``.

    Accepts torch tensors or numpy arrays. Shapes: contact / approach /
    baseline ``(..., 3)``, width ``(...)``.
    """
    half_w = width * 0.5
    if isinstance(contact, torch.Tensor):
        half_w = half_w.unsqueeze(-1)
    else:
        half_w = half_w[..., None]
    return contact + half_w * baseline + PANDA_BASELINE_DIST * approach


def gram_schmidt(
    z1: torch.Tensor, z2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gram-Schmidt orthonormalization (paper Eq. 6).

    Returns (b_hat, a_hat) where b_hat = normalise(z1) and a_hat is the
    component of z2 orthogonal to b_hat, normalised.
    """
    b = F.normalize(z1, p=2, dim=-1)
    a = z2 - (b * z2).sum(dim=-1, keepdim=True) * b
    a = F.normalize(a, p=2, dim=-1)
    return b, a


class CGNHeads(nn.Module):
    """Per-point grasp prediction heads following the CGN paper (Section III.C).

    * Confidence: 1 logit per point.
    * Baseline (b) and approach (a) directions: coupled via in-network
      Gram-Schmidt orthonormalization (Eq. 6).
    * Width: 10 equidistant bins in [0, wmax] with multi-label BCE.
    """

    def __init__(
        self,
        in_channels: int,
        num_width_bins: int = NUM_WIDTH_BINS,
        gripper_width_max: float = GRIPPER_WIDTH_MAX,
    ):
        super().__init__()
        self.num_width_bins = num_width_bins
        self.gripper_width_max = gripper_width_max

        bin_width = gripper_width_max / num_width_bins
        centres = torch.arange(num_width_bins).float() * bin_width + bin_width / 2
        self.register_buffer("bin_centres", centres)

        hidden = max(in_channels, 64)

        self.conv_conf = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, 1, 1),
        )

        self.conv_z1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, 3, 1),
        )

        self.conv_z2 = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, 3, 1),
        )

        self.conv_width = nn.Sequential(
            nn.Conv1d(in_channels, hidden, 1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, num_width_bins, 1),
        )

    def forward(self, features: torch.Tensor) -> Predictions:
        """
        features: (B, C, N) tensor of per-point features from the backbone.
        """
        conf_logits = self.conv_conf(features).squeeze(1)             # (B, N)
        z1 = self.conv_z1(features).transpose(1, 2)                   # (B, N, 3)
        z2 = self.conv_z2(features).transpose(1, 2)                   # (B, N, 3)
        width_bin_logits = self.conv_width(features).transpose(1, 2)  # (B, N, num_bins)

        base_dirs, approach_dirs = gram_schmidt(z1, z2)               # (B, N, 3) each

        best_bin = width_bin_logits.argmax(dim=-1)                    # (B, N)
        widths = self.bin_centres[best_bin]                           # (B, N)

        return Predictions(
            confidence=torch.sigmoid(conf_logits),
            confidence_logits=conf_logits,
            approach_dirs=approach_dirs,
            base_dirs=base_dirs,
            width_bin_logits=width_bin_logits,
            widths=widths,
        )
