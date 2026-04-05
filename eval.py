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
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

import config
from utils.embedding_dataset import EmbeddingDataset
from models.fusion_model import FusionModel

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Comic Emotion Classifier")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (auto-finds latest if omitted)")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir=config.CHECKPOINTS_DIR):
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpts:
        raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")
    return os.path.join(ckpt_dir, ckpts[-1])


def run_inference(dl, model, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in dl:
            img, text, ys = batch
            img, text = img.to(device), text.to(device)

            outputs = model(img, text)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()

            y_true.extend(ys.numpy())
            y_pred.extend(preds)

    return np.array(y_true), np.array(y_pred)


def print_metrics(y_true, y_pred):
    if len(y_true) == 0:
        print("  ⚠  No samples to evaluate.")
        return

    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names  = [config.IDX_TO_LABEL[i] for i in present_labels]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0, labels=present_labels)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0, labels=present_labels)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0, labels=present_labels)

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


def main():
    args   = parse_args()
    device = args.device

    try:
        ckpt_path = args.checkpoint or find_latest_checkpoint()
    except FileNotFoundError as e:
        print(f"\n✘  {e}")
        sys.exit(1)

    print(f"Checkpoint : {ckpt_path}")

    pt_path = "data/embeddings.pt"
    if not os.path.exists(pt_path):
        print(f"\n✘  {pt_path} not found.")
        sys.exit(1)

    try:
        ds = EmbeddingDataset(pt_path, split=args.split)
    except Exception as e:
        print(f"\n✘  Could not load split '{args.split}' from embeddings. {e}")
        sys.exit(1)
        
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    model = FusionModel(n_cls=config.N_CLS).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    y_true, y_pred = run_inference(dl, model, device)

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
