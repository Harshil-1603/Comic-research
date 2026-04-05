"""
Fusion models for Comic Emotion Classification.
Simplified for offline embedding features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class FusionModel(nn.Module):
    """Simple MLP fusion without color features or heavy attention."""
    def __init__(self, d_img=512, d_txt=768, n_cls=None):
        super().__init__()
        n_cls = n_cls or config.N_CLS
        self.net = nn.Sequential(
            nn.Linear(d_img + d_txt, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_cls),
        )

    def forward(self, img, txt):
        x = F.normalize(img, dim=-1)
        z = F.normalize(txt, dim=-1)
        return self.net(torch.cat([x, z], dim=1))
