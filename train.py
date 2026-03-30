import torch, os
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

from data.dataset import ComicDataset
from features.color_features import hsv_hist
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FusionModel
from utils.collate import collate

device = "cuda" if torch.cuda.is_available() else "cpu"

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
])

ds = ComicDataset("data/annotations.csv", "data/processed", transforms=None)
dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate, num_workers=2)

img_enc = ImageEncoder(device)
txt_enc = TextEncoder(device)
model = FusionModel().to(device)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
crit = torch.nn.CrossEntropyLoss()


def to_pil(batch_imgs):
    return [tfm(img) for img in batch_imgs]


for epoch in range(10):
    model.train()
    for imgs, texts, ys in dl:
        pil_imgs = to_pil(imgs)
        img_feat = img_enc(pil_imgs)  # (B,512)
        txt_feat = txt_enc(texts)  # (B,768)
        col_feat = torch.stack([hsv_hist(img) for img in imgs]).to(device)  # (B,48)
        y = torch.tensor(ys).to(device)

        pred = model(img_feat, txt_feat, col_feat)
        loss = crit(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch}.pt")
    print(f"epoch {epoch} loss {loss.item():.4f}")
