"""Shared checkpoint load/save helpers for ``train.py`` and inference.

Training checkpoints store a ``config`` dict (via ``dataclasses.asdict`` on
``TrainConfig``) that includes ``backbone`` and ``cpe_mode``. Prefer those
fields when rebuilding a model; weight-shape heuristics are only a fallback
for older checkpoints that lack an embedded config.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import torch

from models.types import CheckpointConfig

MapLocation = Union[str, torch.device, Mapping[str, str], None]


def torch_load_checkpoint(path: str, map_location: MapLocation = None) -> Any:
    """Load a ``.pt`` file; allow full training dicts (PyTorch 2.6+ safe).

    Training checkpoints embed optimizer/config objects, so ``weights_only``
    cannot be True. Only load files you trust (local training artifacts).
    The opaque return value is narrowed by
    ``state_dict_and_config_from_checkpoint`` / ``load_training_checkpoint``.
    """
    try:
        return torch.load(  # nosec B614
            path, map_location=map_location, weights_only=False
        )
    except TypeError:
        return torch.load(path, map_location=map_location)  # nosec B614


def infer_cpe_mode_from_state_dict(state: dict) -> Optional[str]:
    """Best-effort CPE mode detection from weight shapes/keys.

    Fallback for old checkpoints that have no ``config['cpe_mode']``.
    Prefer reading ``cpe_mode`` from the checkpoint config when present.

    Checks one representative parameter:
      * ``knn``      -> ``backbone.enc_blocks.0.0.cpe.weight`` (bare parameter)
      * ``conv1d`` / ``sparse3d`` -> ``backbone.enc_blocks.0.0.cpe.conv.weight``

    Returns ``None`` if the keys are not found (e.g. ``pn2`` backbone).
    """
    if not isinstance(state, dict):
        return None
    if "backbone.enc_blocks.0.0.cpe.weight" in state:
        return "knn"
    if "backbone.enc_blocks.0.0.cpe.conv.weight" in state:
        w = state["backbone.enc_blocks.0.0.cpe.conv.weight"]
        # sparse3d's SubMConv3d weight is 5-D (kD,kH,kW, Cin, Cout),
        # Conv1DCPE's Conv1d weight is 3-D (Cout, Cin/groups, k).
        if hasattr(w, "ndim"):
            return "sparse3d" if w.ndim >= 4 else "conv1d"
    return None


def state_dict_and_config_from_checkpoint(
    raw: Any,
) -> Tuple[dict, CheckpointConfig]:
    """Split ``torch.load`` output into a ``state_dict`` and optional train config.

    Supports ``train.py`` checkpoints (``model_state_dict`` + ``config``),
    ``{"model": ...}``, Lightning-style ``state_dict``, or a bare state_dict.
    """
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected checkpoint to be a dict, got {type(raw).__name__}"
        )

    ckpt_cfg: CheckpointConfig = {}
    if "model_state_dict" in raw:
        return raw["model_state_dict"], dict(raw.get("config") or {})
    if "model" in raw:
        return raw["model"], dict(raw.get("config") or {})
    if "state_dict" in raw:
        return raw["state_dict"], dict(raw.get("config") or {})

    if any(k in raw for k in ("epoch", "optimizer_state_dict", "scheduler_state_dict")):
        raise ValueError(
            "Checkpoint dict is missing 'model_state_dict' / 'model' / 'state_dict'."
        )
    return raw, ckpt_cfg


def save_training_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    best_val_loss: float,
    config: Mapping[str, Any] | CheckpointConfig,
) -> None:
    """Persist model + optimizer (+ optional scheduler) and the train config.

    ``config`` should include at least ``backbone`` and ``cpe_mode`` so
    inference can rebuild a matching graph without weight heuristics.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": dict(config),
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_training_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: MapLocation = "cpu",
) -> tuple[dict[str, Any], int, float]:
    """Load a training checkpoint into ``model`` (+ optional optimizer/scheduler)."""
    raw = torch_load_checkpoint(path, map_location=device)
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected checkpoint to be a dict, got {type(raw).__name__}"
        )
    state_dict, _cfg = state_dict_and_config_from_checkpoint(raw)
    model.load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in raw:
        optimizer.load_state_dict(raw["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in raw:
        scheduler.load_state_dict(raw["scheduler_state_dict"])

    start_epoch = int(raw.get("epoch", -1)) + 1
    best_val_loss = float(raw.get("best_val_loss", float("inf")))
    return raw, start_epoch, best_val_loss
