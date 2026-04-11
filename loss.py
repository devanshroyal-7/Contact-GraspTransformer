import torch
import torch.nn as nn
import torch.nn.functional as F

class CGNLoss(nn.Module):
    def __init__(self, app_weight=0.1, base_weight=0.1, width_weight=1.0):
        super().__init__()
        self.app_weight = app_weight
        self.base_weight = base_weight
        self.width_weight = width_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, preds, targets):
        """
        preds: dictionary from CGN forward
        targets: dictionary from dataset loader
        """
        conf_logits = preds['confidence_logits'] # (B, N)
        target_conf = targets['confidence'] # (B, N)
        
        # 1. Confidence Loss (BCE on all points)
        loss_conf = self.bce(conf_logits, target_conf)
        
        # Mask for regression losses (only supervise on points with valid grasps)
        mask = target_conf > 0.5
        
        loss_app = torch.tensor(0.0, device=conf_logits.device)
        loss_base = torch.tensor(0.0, device=conf_logits.device)
        loss_width = torch.tensor(0.0, device=conf_logits.device)
        
        if mask.any():
            # Apply mask, flatten to (K, ...)
            pred_app = preds['approach_dirs'][mask] # (K, 3)
            pred_base = preds['base_dirs'][mask] # (K, 3)
            pred_width = preds['widths'][mask] # (K,)
            
            targ_app = targets['approach_dirs'][mask] 
            targ_base = targets['base_dirs'][mask]
            targ_width = targets['widths'][mask]
            
            # 2. Approach Direction Loss (Cosine Similarity or L2)
            # Typically using 1 - cosine similarity
            loss_app = (1.0 - F.cosine_similarity(pred_app, targ_app, dim=-1)).mean()
            
            # 3. Base Direction Loss
            loss_base = (1.0 - F.cosine_similarity(pred_base, targ_base, dim=-1)).mean()
            
            # 4. Width Loss (L1 or L2)
            loss_width = self.l1(pred_width, targ_width).mean()
            
        total_loss = (loss_conf
                      + self.app_weight * loss_app
                      + self.base_weight * loss_base
                      + self.width_weight * loss_width)
        
        loss_dict = {
            'loss': total_loss,
            'l_conf': loss_conf,
            'l_app': loss_app,
            'l_base': loss_base,
            'l_width': loss_width
        }
        return loss_dict
