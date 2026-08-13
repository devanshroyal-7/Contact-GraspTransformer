from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from models.backbone_pn2 import SimplePointNet2
from models.backbone_ptv3 import PTv3Wrapper
from models.cgn_heads import CGNHeads
from models.types import Predictions

_VALID_CPE_MODES = frozenset({"knn", "conv1d", "sparse3d"})


class ContactGraspNet(nn.Module):
    """
    Main model encapsulating the backbone and the heads.
    User can specify backbone='pn2' or backbone='ptv3' for comparison.

    PTv3-only knobs (``cpe_mode``, ``in_channels``, ``window_size``) are typed
    constructor parameters; unknown values are rejected at this boundary rather
    than being **forwarded into the backbone.
    """

    def __init__(
        self,
        backbone: str = "ptv3",
        *,
        cpe_mode: str | None = None,
        in_channels: int | None = None,
        window_size: int | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.head_in_channels = 64

        if backbone == "pn2":
            if cpe_mode is not None or in_channels is not None or window_size is not None:
                raise ValueError(
                    "cpe_mode, in_channels, and window_size apply only to "
                    "backbone='ptv3'"
                )
            self.backbone = SimplePointNet2(out_channels=self.head_in_channels)
        elif backbone == "ptv3":
            ptv3_kwargs = {"out_channels": self.head_in_channels}
            if cpe_mode is not None:
                if cpe_mode not in _VALID_CPE_MODES:
                    raise ValueError(
                        f"Unknown cpe_mode={cpe_mode!r}; "
                        f"expected one of {sorted(_VALID_CPE_MODES)}"
                    )
                ptv3_kwargs["cpe_mode"] = cpe_mode
            if in_channels is not None:
                ptv3_kwargs["in_channels"] = in_channels
            if window_size is not None:
                ptv3_kwargs["window_size"] = window_size
            self.backbone = PTv3Wrapper(**ptv3_kwargs)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.heads = CGNHeads(in_channels=self.head_in_channels)

    def forward(
        self,
        xyz: torch.Tensor,
        feat: Optional[torch.Tensor] = None,
    ) -> Predictions:
        """
        xyz: (B, N, 3) point cloud
        feat: optional (B, N, C_extra) per-point extras (normals, colors, ...).
              Requires backbone='ptv3'. Channel count must satisfy
              ``3 + C_extra == backbone.in_channels`` (default in_channels=3
              means ``feat`` must be omitted; pass ``in_channels`` at
              construction when extras are used).

        Returns ``Predictions`` with keys:
        confidence, confidence_logits, approach_dirs, base_dirs,
        width_bin_logits, widths, and features (backbone per-point features).
        """
        if feat is not None:
            if self.backbone_name != "ptv3":
                raise ValueError(
                    "feat requires backbone='ptv3'; "
                    f"got backbone={self.backbone_name!r}"
                )
            features = self.backbone(xyz, feat)
        else:
            features = self.backbone(xyz)
        preds = self.heads(features)
        preds["features"] = features
        return preds
