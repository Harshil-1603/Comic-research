"""
Fusion models for Comic Emotion Classification.

FusionModel  — MLP baseline (used in ablation study)
AttnFusion   — Cross-attention + colour (main model, used in train.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class FusionModel(nn.Module):
    """MLP baseline — used only in ablation experiments."""
    def __init__(self, d_img=512, d_txt=768, d_col=48, n_cls=None):
        super().__init__()
        n_cls = n_cls or config.N_CLS
        self.net = nn.Sequential(
            nn.Linear(d_img + d_txt + d_col, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_cls),
        )

    def forward(self, img, txt, col):
        x = F.normalize(img, dim=-1)
        z = F.normalize(txt, dim=-1)
        return self.net(torch.cat([x, z, col], dim=1))


class AttnFusion(nn.Module):
    """
    Cross-attention fusion (main model).

    Architecture:
      Q ← linear(img)          (B, 1, d)
      K,V ← linear(txt)        (B, 1, d)
      A ← MultiheadAttention(Q,K,V)  (B, 1, d) → squeeze → (B, d)
      out ← linear(concat(A, col))   (B, d+48) → (B, n_cls)

    Dropout applied before the final classifier for regularisation.
    """
    def __init__(self, d=None, n_cls=None):
        super().__init__()
        d     = d     or config.D_ATTN
        n_cls = n_cls or config.N_CLS

        self.q    = nn.Linear(config.D_IMG, d)
        self.k    = nn.Linear(config.D_TXT, d)
        self.v    = nn.Linear(config.D_TXT, d)
        self.attn = nn.MultiheadAttention(d, num_heads=8, batch_first=True, dropout=0.1)
        self.drop = nn.Dropout(0.3)
        self.fc   = nn.Linear(d + config.D_COL, n_cls)

    def forward(self, img, txt, col):
        """
        Args:
            img: (B, D_IMG)  — CLIP features
            txt: (B, D_TXT)  — BERT features
            col: (B, D_COL)  — HSV histogram features
        Returns:
            logits: (B, N_CLS)
        """
        Q = self.q(img).unsqueeze(1)          # (B, 1, d)
        K = self.k(txt).unsqueeze(1)          # (B, 1, d)
        V = self.v(txt).unsqueeze(1)          # (B, 1, d)
        A, _ = self.attn(Q, K, V)             # (B, 1, d)
        A = A.squeeze(1)                      # (B, d)
        return self.fc(self.drop(torch.cat([A, col], dim=1)))
