"""
Simplified offline embedding training loop for Comic Emotion Classification.

Key changes vs baseline:
  - Embeddings are precomputed (extract_features.py)
  - Uses simple FusionModel (MLP)
  - No GPU OOM problems, trains lightning fast
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils.reproducibility import seed_all, log_config, get_default_config
from utils.class_weights import compute_class_weights
from utils.embedding_dataset import EmbeddingDataset
from models.fusion_model import FusionModel

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Comic Emotion Classifier on Embeddings")
    p.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Fusion head learning rate")
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--split-csv",   type=str,   default=config.SPLIT_CSV)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Train / eval epoch ────────────────────────────────────────────────────────

def run_epoch(model, loader, opt, crit, device, training=True):
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    grad_ctx = torch.enable_grad() if training else torch.no_grad()

    with grad_ctx:
        for step, batch in enumerate(loader):
            img, text, y = batch
            img, text, y = img.to(device), text.to(device), y.to(device)

            if training and step == 0:
                print(f"    [Debug] Batch shape: Img={list(img.shape)}, Txt={list(text.shape)}")

            pred = model(img, text)             # (B, N_CLS)
            loss = crit(pred, y)

            if training:
                loss.backward()
                opt.step()
                opt.zero_grad()

            total_loss += loss.item()
            correct    += (pred.argmax(1) == y).sum().item()
            total      += y.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, opt):
    torch.save({
        "epoch":   epoch,
        "model":   model.state_dict(),
        "opt":     opt.state_dict(),
        "config":  {"n_cls": config.N_CLS, "label_map": config.LABEL_MAP},
    }, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    seed_all(config.SEED)
    device  = args.device

    csv_path  = args.split_csv if os.path.exists(args.split_csv) else config.ANNOTATIONS_CSV

    print(f"\nDevice    : {device}")
    print(f"CSV       : {csv_path}")
    print(f"N classes : {config.N_CLS}  → {list(config.LABEL_MAP.keys())}")
    print(f"LR        : {args.lr}")

    # ── Log config ────────────────────────────────────────────────────────────
    cfg = get_default_config()
    cfg.update({
        "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size,
        "model": "FusionModel (Embeddings)", "fine_tune_encoders": False,
    })
    log_config(cfg, log_dir=config.LOGS_DIR)

    # ── Data loaders ──────────────────────────────────────────────────────────
    pt_path = "data/embeddings.pt"
    if not os.path.exists(pt_path):
        print(f"ERROR: {pt_path} not found. Run extract_features.py first.")
        return

    def make_loader(split_name, shuffle):
        try:
            ds = EmbeddingDataset(pt_path, split=split_name)
            return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)
        except Exception as e:
            print(f"Warning: split '{split_name}' not formed correctly. {e}")
            return None

    train_loader = make_loader("train", shuffle=True)
    val_loader   = make_loader("val", shuffle=False)
    
    if not train_loader:
        print("Falling back to generic split loading")
        train_loader = make_loader("all", shuffle=True)
        val_loader = None

    print(f"Train set : {len(train_loader.dataset)} samples")
    if val_loader:
        print(f"Val set   : {len(val_loader.dataset)} samples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FusionModel(n_cls=config.N_CLS).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-7
    )

    # ── Weighted loss ─────────────────────────────────────────────────────────
    # Keeping class labels logic exactly the same, but using train split directly
    class_w = compute_class_weights(csv_path, split="train", device=device)
    crit    = nn.CrossEntropyLoss(weight=class_w)
    print(f"\nUsing weighted loss + simplified FusionModel")

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(
            model, train_loader, opt, crit, device, training=True,
        )
        line = (f"epoch {epoch:02d}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}")

        if val_loader:
            va_loss, va_acc = run_epoch(
                model, val_loader, opt, crit, device, training=False,
            )
            line += f"  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}"

        scheduler.step()
        print(line)

        ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f"epoch_{epoch:02d}.pt")
        save_checkpoint(ckpt_path, epoch, model, opt)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
