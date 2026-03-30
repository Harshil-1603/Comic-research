import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    """MLP-based fusion model"""
    def __init__(self, d_img=512, d_txt=768, d_col=48, n_cls=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_img + d_txt + d_col, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cls)
        )

    def forward(self, img, txt, col):
        x = F.normalize(img, dim=-1)
        z = F.normalize(txt, dim=-1)
        c = col
        fused = torch.cat([x, z, c], dim=1)
        return self.net(fused)


class AttnFusion(nn.Module):
    """Attention-based multimodal fusion"""
    def __init__(self, d=512, n_cls=5):
        super().__init__()
        self.q = nn.Linear(512, d)
        self.k = nn.Linear(768, d)
        self.v = nn.Linear(768, d)
        self.attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
        self.fc = nn.Linear(d + 48, n_cls)

    def forward(self, img, txt, col):
        Q = self.q(img).unsqueeze(1)   # (B,1,d)
        K = self.k(txt).unsqueeze(1)   # (B,1,d)
        V = self.v(txt).unsqueeze(1)
        A, _ = self.attn(Q, K, V)      # (B,1,d)
        A = A.squeeze(1)
        return self.fc(torch.cat([A, col], dim=1))
