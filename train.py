"""
Fine-tuning training loop for Comic Emotion Classification.

Key changes vs baseline:
  - Uses AttnFusion (cross-attention) instead of MLP
  - Fine-tunes CLIP + BERT with differential learning rates
  - AMP (fp16) + gradient checkpointing for 6GB VRAM
  - Weighted CrossEntropyLoss for class imbalance
  - Saves full checkpoint (model + encoders) for eval/inference

Usage:
    python train.py
    python train.py --epochs 20 --device cuda
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from utils.reproducibility import seed_all, log_config, get_default_config
from utils.class_weights import compute_class_weights
from data.dataset import ComicDataset
from features.color_features import hsv_hist
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import AttnFusion
from utils.collate import collate

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Comic Emotion Classifier")
    p.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    p.add_argument("--lr",          type=float, default=config.LEARNING_RATE,
                   help="Fusion head learning rate")
    p.add_argument("--encoder-lr",  type=float, default=config.ENCODER_LR,
                   help="Encoder (CLIP+BERT) learning rate")
    p.add_argument("--batch-size",  type=int,   default=config.BATCH_SIZE)
    p.add_argument("--split-csv",   type=str,   default=config.SPLIT_CSV)
    p.add_argument("--device",      type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp",      action="store_true",
                   help="Disable automatic mixed precision")
    return p.parse_args()


# ── Image preprocessing ───────────────────────────────────────────────────────

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
])


def to_pil(batch_imgs):
    return [tfm(img) for img in batch_imgs]


# ── Train / eval epoch ────────────────────────────────────────────────────────

def run_epoch(model, img_enc, txt_enc, loader,
              opt, scaler, crit, device, use_amp, training=True):

    # Set all modules to correct mode
    for m in [model, img_enc, txt_enc]:
        m.train() if training else m.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_params = (list(model.parameters())
                  + list(img_enc.parameters())
                  + list(txt_enc.parameters()))

    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for imgs, texts, ys in loader:
            pil_imgs = to_pil(imgs)
            y        = ys.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                img_feat = img_enc(pil_imgs)                           # (B, 512)
                txt_feat = txt_enc(texts)                              # (B, 768)
                col_feat = torch.stack(
                    [hsv_hist(img) for img in imgs]
                ).to(device)                                           # (B, 48)
                pred = model(img_feat, txt_feat, col_feat)             # (B, N_CLS)
                loss = crit(pred, y)

            if training:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                # Gradient clipping prevents exploding gradients during fine-tuning
                nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                scaler.step(opt)
                scaler.update()

            total_loss += loss.item()
            correct    += (pred.argmax(1) == y).sum().item()
            total      += y.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, img_enc, txt_enc, opt):
    torch.save({
        "epoch":   epoch,
        "model":   model.state_dict(),
        "img_enc": img_enc.state_dict(),
        "txt_enc": txt_enc.state_dict(),
        "opt":     opt.state_dict(),
        "config":  {"n_cls": config.N_CLS, "label_map": config.LABEL_MAP},
    }, path)


def load_checkpoint(path, model, img_enc, txt_enc, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "img_enc" in ckpt: img_enc.load_state_dict(ckpt["img_enc"])
        if "txt_enc" in ckpt: txt_enc.load_state_dict(ckpt["txt_enc"])
        return ckpt.get("epoch", -1)
    else:
        # Backward compat: old format was just model state_dict
        model.load_state_dict(ckpt)
        return -1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    seed_all(config.SEED)
    device  = args.device
    use_amp = config.USE_AMP and not args.no_amp and device == "cuda"

    # ── CSV / split ───────────────────────────────────────────────────────────
    csv_path  = args.split_csv if os.path.exists(args.split_csv) else config.ANNOTATIONS_CSV
    has_split = False
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        has_split = "split" in df.columns
    except Exception:
        pass

    split     = "train" if has_split else None
    val_split = "val"   if has_split else None

    print(f"\nDevice    : {device}  (AMP={'ON' if use_amp else 'OFF'})")
    print(f"CSV       : {csv_path}  (split='{split}')")
    print(f"N classes : {config.N_CLS}  → {list(config.LABEL_MAP.keys())}")
    print(f"Encoder LR: {args.encoder_lr}   Fusion LR: {args.lr}")

    # ── Log config ────────────────────────────────────────────────────────────
    cfg = get_default_config()
    cfg.update({
        "epochs": args.epochs, "lr": args.lr,
        "encoder_lr": args.encoder_lr, "batch_size": args.batch_size,
        "model": "AttnFusion", "fine_tune_encoders": True,
    })
    log_config(cfg, log_dir=config.LOGS_DIR)

    # ── Data loaders ──────────────────────────────────────────────────────────
    def make_loader(split_name, shuffle):
        ds = ComicDataset(csv_path, config.PROCESSED_DIR, split=split_name)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=config.NUM_WORKERS)

    train_loader = make_loader(split, shuffle=True)
    val_loader   = make_loader(val_split, shuffle=False) if val_split else None

    print(f"Train set : {len(train_loader.dataset)} samples")
    if val_loader:
        print(f"Val set   : {len(val_loader.dataset)} samples")

    # ── Models ────────────────────────────────────────────────────────────────
    img_enc = ImageEncoder(device)
    txt_enc = TextEncoder(device)
    model   = AttnFusion(d=config.D_ATTN, n_cls=config.N_CLS).to(device)

    # ── Differential LR optimizer ─────────────────────────────────────────────
    # Encoders get a 10× smaller LR to avoid catastrophic forgetting of
    # pre-trained CLIP/BERT representations
    opt = torch.optim.AdamW([
        {"params": img_enc.parameters(), "lr": args.encoder_lr, "weight_decay": 1e-4},
        {"params": txt_enc.parameters(), "lr": args.encoder_lr, "weight_decay": 1e-4},
        {"params": model.parameters(),   "lr": args.lr,         "weight_decay": 1e-4},
    ])

    # LR scheduler — cosine decay over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-7
    )

    # ── Weighted loss ─────────────────────────────────────────────────────────
    class_w = compute_class_weights(csv_path, split=split, device=device)
    crit    = nn.CrossEntropyLoss(weight=class_w)
    print(f"\nUsing weighted loss + AttnFusion + fine-tuned CLIP & BERT")

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    print()
    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(
            model, img_enc, txt_enc, train_loader,
            opt, scaler, crit, device, use_amp, training=True,
        )
        line = (f"epoch {epoch:02d}  "
                f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}")

        if val_loader:
            va_loss, va_acc = run_epoch(
                model, img_enc, txt_enc, val_loader,
                opt, scaler, crit, device, use_amp, training=False,
            )
            line += f"  val_loss={va_loss:.4f}  val_acc={va_acc:.4f}"

        scheduler.step()
        print(line)

        ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f"epoch_{epoch:02d}.pt")
        save_checkpoint(ckpt_path, epoch, model, img_enc, txt_enc, opt)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
