import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
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
