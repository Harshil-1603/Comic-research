"""
Evaluation script — Accuracy, Precision, Recall, F1 (macro), Confusion Matrix.
Matches plan §0.3 metrics.

Designed to be crash-proof:
  - Handles any subset of the 5 emotion classes in the test split
  - Falls back gracefully if dataset is empty, checkpoint missing, etc.

Usage:
    python eval.py
    python eval.py --split test --checkpoint checkpoints/epoch_09.pt
"""
import os
import sys
import argparse
import traceback
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

import config
from data.dataset import ComicDataset, label_map
from features.color_features import hsv_hist
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import AttnFusion
from utils.collate import collate

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Arg parsing ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Comic Emotion Classifier")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (auto-finds latest if omitted)")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--split-csv", type=str, default=config.SPLIT_CSV)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir=config.CHECKPOINTS_DIR):
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpts:
        raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")
    return os.path.join(ckpt_dir, ckpts[-1])


tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
])


def to_pil(batch_imgs):
    return [tfm(img) for img in batch_imgs]


# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(dl, model, img_enc, txt_enc, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, texts, ys in dl:
            pil_imgs = to_pil(imgs)
            img_feat = img_enc(pil_imgs)
            txt_feat = txt_enc(texts)
            col_feat = torch.stack([hsv_hist(img) for img in imgs]).to(device)

            outputs = model(img_feat, txt_feat, col_feat)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(ys.numpy())
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


# ── Metrics ───────────────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred):
    """
    Print all plan §0.3 metrics.
    Works for any subset of the 5 emotion classes — never crashes on a
    class-count mismatch.
    """
    if len(y_true) == 0:
        print("  ⚠  No samples to evaluate.")
        return

    # Only use labels actually present (avoids sklearn ValueError)
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names  = [config.IDX_TO_LABEL[i] for i in present_labels]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro",
                           zero_division=0, labels=present_labels)
    rec  = recall_score(y_true, y_pred, average="macro",
                        zero_division=0, labels=present_labels)
    f1   = f1_score(y_true, y_pred, average="macro",
                    zero_division=0, labels=present_labels)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Classes present  : {present_names}")
    print(f"Total samples    : {len(y_true)}")
    print(f"Accuracy         : {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall    (macro): {rec:.4f}")
    print(f"F1        (macro): {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=present_labels))

    return present_labels, present_names


def save_confusion_matrix(y_true, y_pred, present_labels, present_names):
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    n = len(present_names)
    ax.set_xticks(range(n))
    ax.set_xticklabels(present_names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(present_names)

    # Cell value annotations
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > vmax / 2 else "black",
                    fontsize=11, fontweight="bold")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    cm_path = os.path.join(config.LOGS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {cm_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = args.device

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    try:
        ckpt_path = args.checkpoint or find_latest_checkpoint()
    except FileNotFoundError as e:
        print(f"\n✘  {e}")
        print("   Train the model first:  python train.py")
        sys.exit(1)

    print(f"Checkpoint : {ckpt_path}")

    # ── Resolve CSV / split ───────────────────────────────────────────────────
    import pandas as pd
    csv_path = args.split_csv if os.path.exists(args.split_csv) else config.ANNOTATIONS_CSV

    if not os.path.exists(csv_path):
        print(f"\n✘  Annotations CSV not found: {csv_path}")
        sys.exit(1)

    df    = pd.read_csv(csv_path)
    split = args.split if "split" in df.columns else None
    print(f"CSV        : {csv_path}  (split='{split}')")

    # ── Dataset & loader ─────────────────────────────────────────────────────
    ds = ComicDataset(csv_path, config.PROCESSED_DIR, split=split)
    if len(ds) == 0:
        print(f"\n✘  No samples found for split='{split}'. "
              "Run utils/split_data.py to regenerate splits.")
        sys.exit(1)

    dl = DataLoader(ds, batch_size=16, shuffle=False,
                    collate_fn=collate, num_workers=config.NUM_WORKERS)

    # ── Load model & encoders ─────────────────────────────────────────────────
    model   = AttnFusion(d=config.D_ATTN, n_cls=config.N_CLS).to(device)
    img_enc = ImageEncoder(device)
    txt_enc = TextEncoder(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "img_enc" in ckpt: img_enc.load_state_dict(ckpt["img_enc"])
        if "txt_enc" in ckpt: txt_enc.load_state_dict(ckpt["txt_enc"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    img_enc.eval()
    txt_enc.eval()

    # ── Run inference ─────────────────────────────────────────────────────────
    y_true, y_pred = run_inference(dl, model, img_enc, txt_enc, device)

    # ── Print metrics (never crashes regardless of class subset) ──────────────
    result = print_metrics(y_true, y_pred)
    if result:
        present_labels, present_names = result
        try:
            save_confusion_matrix(y_true, y_pred, present_labels, present_names)
        except Exception as e:
            print(f"  ⚠  Could not save confusion matrix: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n✘  eval.py encountered an unexpected error:")
        traceback.print_exc()
        sys.exit(1)
