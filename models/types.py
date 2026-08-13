"""Shared typed contracts for samples, predictions, and loss outputs.

Dict conversion stays at npz / checkpoint boundaries; training and model code
should use these TypedDict shapes instead of ad-hoc string-key bags.
"""

from __future__ import annotations

from typing import TypedDict

try:
    from typing_extensions import NotRequired
except ImportError:  # pragma: no cover
    from typing import NotRequired  # Python 3.11+

import numpy as np
import torch


class Predictions(TypedDict):
    """Output of ``CGNHeads`` / ``ContactGraspNet.forward``."""

    confidence: torch.Tensor
    confidence_logits: torch.Tensor
    approach_dirs: torch.Tensor
    base_dirs: torch.Tensor
    width_bin_logits: torch.Tensor
    widths: torch.Tensor
    features: NotRequired[torch.Tensor]


class SampleBatch(TypedDict):
    """One training sample from ``CGNDataset.__getitem__`` (torch tensors)."""

    points: torch.Tensor
    confidence: torch.Tensor
    approach_dirs: torch.Tensor
    base_dirs: torch.Tensor
    widths: torch.Tensor


class GraspSampleNP(TypedDict):
    """Numpy grasp sample consumed by ``viz.dataset_visualizer`` helpers."""

    points: np.ndarray
    confidence: np.ndarray
    approach_dirs: np.ndarray
    base_dirs: np.ndarray
    widths: np.ndarray


class LossBreakdown(TypedDict):
    """Scalar loss terms returned by ``CGNLoss.forward``."""

    loss: torch.Tensor
    l_conf: torch.Tensor
    l_adds: torch.Tensor
    l_width: torch.Tensor


class CheckpointConfig(TypedDict, total=False):
    """Train config embedded in ``.pt`` checkpoints for rebuild/resume.

    Required for reliable inference rebuild: ``backbone``, ``cpe_mode``,
    ``num_points``. Other ``TrainConfig`` fields may be present via
    ``dataclasses.asdict``.
    """

    backbone: str
    cpe_mode: str
    num_points: int
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    scheduler: str
    scheduler_gamma: float
    grad_clip_max_norm: float
    loss_adds_weight: float
    loss_width_weight: float
    data_dir: str
    manifest: str
    budgets: str
    checkpoint_dir: str
    resume: str | None
    wandb_project: str
    wandb_entity: str
    wandb_mode: str
