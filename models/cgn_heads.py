import torch
import torch.nn as nn
import torch.nn.functional as F

class CGNHeads(nn.Module):
    """
    MLP heads predicting grasp confidence, approach direction, base direction, and width for each point.
    Expects input features of shape (B, C, N).
    """
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv_conf = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1) # Output 1 confidence per point
        )
        
        self.conv_app = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1) # Approach direction (3D vector)
        )
        
        self.conv_base = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 3, 1) # Base direction (3D vector)
        )
        
        self.conv_width = nn.Sequential(
            nn.Conv1d(in_channels, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1) # Grasp width scalar
        )
        
    def forward(self, features):
        """
        features: (B, C, N) tensor of per-point features from the backbone.
        """
        conf_logits = self.conv_conf(features).squeeze(1) # (B, N)
        app_dirs = self.conv_app(features).transpose(1, 2) # (B, N, 3)
        base_dirs = self.conv_base(features).transpose(1, 2) # (B, N, 3)
        widths = self.conv_width(features).squeeze(1) # (B, N)
        
        # Normalize direction vectors
        app_dirs = F.normalize(app_dirs, p=2, dim=-1)
        base_dirs = F.normalize(base_dirs, p=2, dim=-1)
        
        return {
            'confidence': torch.sigmoid(conf_logits),
            'confidence_logits': conf_logits,
            'approach_dirs': app_dirs,
            'base_dirs': base_dirs,
            'widths': widths
        }
