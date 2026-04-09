import torch
import torch.nn as nn
from models.backbone_pn2 import SimplePointNet2
from models.backbone_ptv3 import PTv3Wrapper
from models.cgn_heads import CGNHeads

class ContactGraspNet(nn.Module):
    """
    Main model encapsulating the backbone and the heads.
    User can specify backbone='pn2' or backbone='ptv3' for comparison.
    """
    def __init__(self, backbone_type='pn2'):
        super().__init__()
        self.head_in_channels = 128
        
        if backbone_type == 'pn2':
            self.backbone = SimplePointNet2(out_channels=self.head_in_channels)
        elif backbone_type == 'ptv3':
            self.backbone = PTv3Wrapper(out_channels=self.head_in_channels)
        else:
            raise ValueError(f"Unknown backbone: {backbone_type}")
            
        self.heads = CGNHeads(in_channels=self.head_in_channels)
        
    def forward(self, xyz):
        """
        xyz: (B, N, 3) point cloud
        Returns dictionary of predictions: confidence, approach_dirs, base_dirs, widths
        """
        features = self.backbone(xyz) # (B, C, N)
        preds = self.heads(features)
        
        # Depending on evaluation needs, sometimes we append the features:
        preds['features'] = features
        
        return preds
