import torch
import torch.nn as nn
from models.backbone_pn2 import SimplePointNet2
from models.backbone_ptv3 import PTv3Wrapper
from models.cgn_heads import CGNHeads

class ContactGraspNet(nn.Module):
    """
    Main model encapsulating the backbone and the heads.
    User can specify backbone='pn2' or backbone='ptv3' for comparison.

    ``backbone_kwargs`` are forwarded to the backbone constructor; use this to
    pick ``in_channels`` (3 = xyz-only, 6 = xyz + normals/colors, ...), swap
    ``cpe_mode`` between ``"knn" / "conv1d" / "sparse3d"``, or tune
    ``window_size`` / depths without touching this file.
    """
    def __init__(self, backbone_type='pn2', backbone_kwargs=None):
        super().__init__()
        self.head_in_channels = 64
        backbone_kwargs = dict(backbone_kwargs or {})

        if backbone_type == 'pn2':
            self.backbone = SimplePointNet2(
                out_channels=self.head_in_channels, **backbone_kwargs
            )
        elif backbone_type == 'ptv3':
            self.backbone = PTv3Wrapper(
                out_channels=self.head_in_channels, **backbone_kwargs
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")

        self.heads = CGNHeads(in_channels=self.head_in_channels)

    def forward(self, xyz, feat=None):
        """
        xyz: (B, N, 3) point cloud
        feat: optional (B, N, C_extra) per-point features (normals, colors, ...)
              forwarded to the backbone if it is configured with
              ``in_channels > 3``.
        Returns dictionary of predictions: confidence, approach_dirs, base_dirs, widths
        """
        if feat is not None:
            features = self.backbone(xyz, feat)
        else:
            features = self.backbone(xyz)
        preds = self.heads(features)

        # Depending on evaluation needs, sometimes we append the features:
        preds['features'] = features

        return preds
