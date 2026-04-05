"""
Ablation Study — Run 3 configurations:
1. Image only (masks text)
2. Text only (masks image)
3. Image + Text

Results logged to experiments/results.csv.

Usage:
    python experiments/run_ablations.py [--epochs 5]
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import config
from utils.reproducibility import seed_all
from utils.embedding_dataset import EmbeddingDataset
from models.fusion_model import FusionModel
import argparse


def make_loader(pt_path, split, batch_size):
    try:
        ds = EmbeddingDataset(pt_path, split=split)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)
    except Exception as e:
        print(f"Warning: could not load split '{split}'. {e}")
        return None

# ── Single experiment ─────────────────────────────────────────────────────

def run_experiment(config_name, use_image, use_text, epochs,
                   train_loader, val_loader, device):
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  use_image={use_image}  use_text={use_text}  epochs={epochs}")
    print(f"{'='*60}")

    seed_all(config.SEED)
    model = FusionModel(n_cls=config.N_CLS).to(device)
    opt  = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Keeping uniform weights for simplicity since the train.py implements complex
    # logic, we'll just weight by frequency across train subset:
    counts = dict.fromkeys(range(config.N_CLS), 0)
    for _, _, label in train_loader.dataset:
        counts[label.item()] += 1
    
    total = sum(counts.values())
    weights = torch.zeros(config.N_CLS, dtype=torch.float32)
    for i in range(config.N_CLS):
        if counts[i] > 0:
            weights[i] = total / (config.N_CLS * counts[i])
    
    crit = nn.CrossEntropyLoss(weight=weights.to(device))

    # ── Training ────────────────────────────────────────────────────────────
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for img, text, y in train_loader:
            img = img.to(device)
            text = text.to(device)
            y = y.to(device)

            if not use_image:
                img = torch.zeros_like(img).to(device)
            if not use_text:
                text = torch.zeros_like(text).to(device)

            loss = crit(model(img, text), y)
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
        for img, text, y in eval_loader:
            img = img.to(device)
            text = text.to(device)

            if not use_image:
                img = torch.zeros_like(img).to(device)
            if not use_text:
                text = torch.zeros_like(text).to(device)

            preds = model(img, text).argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    present = sorted(set(y_true) | set(y_pred))

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=present)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=present)

    print(f"\n  Results: acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

    return {
        "config": config_name,
        "use_image": use_image,
        "use_text":  use_text,
        "accuracy":  acc,
        "precision_macro": prec,
        "recall_macro":    rec,
        "f1_macro":        f1,
    }

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Ablation study using Embeddings")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = args.device
    epochs = args.epochs

    pt_path = "data/embeddings.pt"
    if not os.path.exists(pt_path):
        print(f"ERROR: {pt_path} not found.")
        return

    train_loader = make_loader(pt_path, "train", config.BATCH_SIZE)
    val_loader   = make_loader(pt_path, "val", config.BATCH_SIZE)
    if not train_loader:
        train_loader = make_loader(pt_path, "all", config.BATCH_SIZE)

    results = []
    configs = [
        ("Image Only",       True,  False),
        ("Text Only",        False, True),
        ("Image + Text",     True,  True),
    ]

    for name, use_img, use_text in configs:
        r = run_experiment(name, use_img, use_text, epochs,
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
