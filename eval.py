import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from data.dataset import ComicDataset, label_map
from features.color_features import hsv_hist
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FusionModel
from utils.collate import collate

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = FusionModel().to(device)
model.load_state_dict(torch.load("checkpoints/epoch_9.pt", map_location=device))
model.eval()

# Load encoders
img_enc = ImageEncoder(device)
txt_enc = TextEncoder(device)

# Transforms
tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
])


def to_pil(batch_imgs):
    return [tfm(img) for img in batch_imgs]


# Load validation dataset
ds = ComicDataset("data/annotations.csv", "data/processed", transforms=None)
# Note: In practice, split into train/val/test first
dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate, num_workers=2)

y_true = []
y_pred = []

with torch.no_grad():
    for imgs, texts, ys in dl:
        pil_imgs = to_pil(imgs)
        img_feat = img_enc(pil_imgs)
        txt_feat = txt_enc(texts)
        col_feat = torch.stack([hsv_hist(img) for img in imgs]).to(device)
        
        outputs = model(img_feat, txt_feat, col_feat)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        y_true.extend(ys)
        y_pred.extend(preds)

# Convert to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Print metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=list(label_map.keys())))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
