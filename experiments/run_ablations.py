"""
Ablation Study Experiments
Run 4 configs:
1. Image only
2. Image + Text
3. Image + Color
4. Image + Text + Color

Log to experiments/results.csv
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.dataset import ComicDataset
from features.color_features import hsv_hist
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FusionModel
from utils.collate import collate


device = "cuda" if torch.cuda.is_available() else "cpu"


class AblationFusionModel(nn.Module):
    """Fusion model with configurable modalities"""
    def __init__(self, use_text=True, use_color=True, d_img=512, d_txt=768, d_col=48, n_cls=5):
        super().__init__()
        self.use_text = use_text
        self.use_color = use_color
        
        input_dim = d_img
        if use_text:
            input_dim += d_txt
        if use_color:
            input_dim += d_col
            
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cls)
        )
    
    def forward(self, img, txt, col):
        import torch.nn.functional as F
        x = F.normalize(img, dim=-1)
        parts = [x]
        if self.use_text:
            z = F.normalize(txt, dim=-1)
            parts.append(z)
        if self.use_color:
            parts.append(col)
        fused = torch.cat(parts, dim=1)
        return self.net(fused)


def run_experiment(config_name, use_text, use_color, epochs=5):
    """Run a single ablation experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"Use Text: {use_text}, Use Color: {use_color}")
    print(f"{'='*60}")
    
    # Transforms
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
    ])
    
    def to_pil(batch_imgs):
        return [tfm(img) for img in batch_imgs]
    
    # Load data
    ds = ComicDataset("data/annotations.csv", "data/processed", transforms=None)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate, num_workers=0)
    
    # Initialize encoders and model
    img_enc = ImageEncoder(device)
    txt_enc = TextEncoder(device) if use_text else None
    model = AblationFusionModel(use_text=use_text, use_color=use_color).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for imgs, texts, ys in dl:
            pil_imgs = to_pil(imgs)
            img_feat = img_enc(pil_imgs)
            
            if use_text:
                txt_feat = txt_enc(texts)
            else:
                txt_feat = torch.zeros(img_feat.size(0), 768).to(device)
            
            if use_color:
                col_feat = torch.stack([hsv_hist(img) for img in imgs]).to(device)
            else:
                col_feat = torch.zeros(img_feat.size(0), 48).to(device)
            
            y = torch.tensor(ys).to(device)
            
            pred = model(img_feat, txt_feat, col_feat)
            loss = crit(pred, y)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {epoch_loss/len(dl):.4f}")
    
    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, texts, ys in dl:
            pil_imgs = to_pil(imgs)
            img_feat = img_enc(pil_imgs)
            
            if use_text:
                txt_feat = txt_enc(texts)
            else:
                txt_feat = torch.zeros(img_feat.size(0), 768).to(device)
            
            if use_color:
                col_feat = torch.stack([hsv_hist(img) for img in imgs]).to(device)
            else:
                col_feat = torch.zeros(img_feat.size(0), 48).to(device)
            
            outputs = model(img_feat, txt_feat, col_feat)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            y_true.extend(ys)
            y_pred.extend(preds)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\nResults: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    return {
        'config': config_name,
        'use_text': use_text,
        'use_color': use_color,
        'accuracy': acc,
        'f1_macro': f1
    }


def main():
    """Run all ablation experiments"""
    results = []
    
    # Config 1: Image only
    results.append(run_experiment('Image Only', use_text=False, use_color=False))
    
    # Config 2: Image + Text
    results.append(run_experiment('Image + Text', use_text=True, use_color=False))
    
    # Config 3: Image + Color
    results.append(run_experiment('Image + Color', use_text=False, use_color=True))
    
    # Config 4: Image + Text + Color (Full)
    results.append(run_experiment('Image + Text + Color', use_text=True, use_color=True))
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('experiments/results.csv', index=False)
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("\nResults saved to experiments/results.csv")


if __name__ == "__main__":
    main()
