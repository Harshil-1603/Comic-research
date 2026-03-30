"""
Ablation Study — plan §14: Run 4 configurations.
1. Image only
2. Image + Text
3. Image + Color
4. Image + Text + Color

Results logged to experiments/results.csv.

Usage:
    python experiments/run_ablations.py [--epochs 5]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from utils.reproducibility import seed_all
from data.dataset import ComicDataset
from features.color_features import hsv_hist
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils.collate import collate

import argparse


# ── Ablation-specific fusion model ──────────────────────────────────────────

class AblationFusionModel(nn.Module):
    """MLP fusion with configurable active modalities."""

    def __init__(self, use_text=True, use_color=True,
                 d_img=512, d_txt=768, d_col=48, n_cls=None):
        super().__init__()
        n_cls = n_cls or config.N_CLS
        self.use_text  = use_text
        self.use_color = use_color

        input_dim = d_img
        if use_text:  input_dim += d_txt
        if use_color: input_dim += d_col

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cls),
        )

    def forward(self, img, txt, col):
        parts = [F.normalize(img, dim=-1)]
        if self.use_text:  parts.append(F.normalize(txt, dim=-1))
        if self.use_color: parts.append(col)
        return self.net(torch.cat(parts, dim=1))


# ── Helpers ──────────────────────────────────────────────────────────────────

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
])


def to_pil(batch_imgs):
    return [tfm(img) for img in batch_imgs]


def make_loader(csv_path, split, img_dir, batch_size):
    import pandas as pd
    df = pd.read_csv(csv_path)
    has_split = "split" in df.columns
    kw = dict(split=split) if has_split and split else {}
    ds = ComicDataset(csv_path, img_dir, **kw)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      collate_fn=collate, num_workers=0)


# ── Single experiment ─────────────────────────────────────────────────────

def run_experiment(config_name, use_text, use_color, epochs,
                   train_loader, val_loader, device):
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  use_text={use_text}  use_color={use_color}  epochs={epochs}")
    print(f"{'='*60}")

    seed_all(config.SEED)

    img_enc = ImageEncoder(device)
    txt_enc = TextEncoder(device) if use_text else None
    model   = AblationFusionModel(use_text=use_text, use_color=use_color).to(device)

    opt  = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Calculate weights dynamically from train_loader's dataframe to account for class imbalance
    df = train_loader.dataset.df
    counts = df["emotion"].value_counts()
    weights = torch.zeros(config.N_CLS, dtype=torch.float32)
    for emotion, idx in config.LABEL_MAP.items():
        if counts.get(emotion, 0) > 0:
            weights[idx] = len(df) / (config.N_CLS * counts[emotion])
    
    crit = nn.CrossEntropyLoss(weight=weights.to(device))

    # ── Training ────────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for imgs, texts, ys in train_loader:
            pil_imgs = to_pil(imgs)
            img_feat = img_enc(pil_imgs)

            txt_feat = (txt_enc(texts) if use_text
                        else torch.zeros(img_feat.size(0), 768, device=device))
            col_feat = (torch.stack([hsv_hist(img) for img in imgs]).to(device) if use_color
                        else torch.zeros(img_feat.size(0), 48, device=device))

            y = ys.to(device)           # already a stacked LongTensor from collate

            loss = crit(model(img_feat, txt_feat, col_feat), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        print(f"  epoch {epoch}: loss={epoch_loss/max(len(train_loader),1):.4f}")

    # ── Evaluation on val split ──────────────────────────────────────────────
    model.eval()
    y_true, y_pred = [], []
    eval_loader = val_loader if val_loader else train_loader

    with torch.no_grad():
        for imgs, texts, ys in eval_loader:
            pil_imgs = to_pil(imgs)
            img_feat = img_enc(pil_imgs)

            txt_feat = (txt_enc(texts) if use_text
                        else torch.zeros(img_feat.size(0), 768, device=device))
            col_feat = (torch.stack([hsv_hist(img) for img in imgs]).to(device) if use_color
                        else torch.zeros(img_feat.size(0), 48, device=device))

            preds = model(img_feat, txt_feat, col_feat).argmax(1).cpu().numpy()
            y_true.extend(ys.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Use only labels present in this experiment's data (avoids sklearn ValueError)
    present = sorted(set(y_true) | set(y_pred))

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=present)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=present)

    print(f"\n  Results: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

    return {
        "config": config_name,
        "use_text":  use_text,
        "use_color": use_color,
        "accuracy":  acc,
        "precision_macro": prec,
        "recall_macro":    rec,
        "f1_macro":        f1,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ablation study")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    epochs = args.epochs

    # Use split CSV if available
    csv_path   = config.SPLIT_CSV if os.path.exists(config.SPLIT_CSV) else config.ANNOTATIONS_CSV
    df         = pd.read_csv(csv_path)
    has_split  = "split" in df.columns

    train_loader = make_loader(csv_path, "train" if has_split else None,
                               config.PROCESSED_DIR, config.BATCH_SIZE)
    val_loader   = (make_loader(csv_path, "val", config.PROCESSED_DIR, config.BATCH_SIZE)
                    if has_split else None)

    results = []
    configs = [
        ("Image Only",          False, False),
        ("Image + Text",        True,  False),
        ("Image + Color",       False, True),
        ("Image + Text + Color", True,  True),
    ]

    for name, use_text, use_color in configs:
        r = run_experiment(name, use_text, use_color, epochs,
                           train_loader, val_loader, device)
        results.append(r)

    # Save results
    os.makedirs(config.EXPERIMENTS_DIR, exist_ok=True)
    results_path = os.path.join(config.EXPERIMENTS_DIR, "results.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_path, index=False)

    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
