"""
Inference script for Comic Emotion Classification — plan §19.

Load encoders + fusion model, preprocess one image, print predicted class.

Usage:
    python inference.py path/to/panel.jpg
    python inference.py path/to/panel.jpg --text "I hate this!"
    python inference.py path/to/panel.jpg --checkpoint checkpoints/epoch_09.pt
"""
import sys
import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image

import config
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from models.fusion_model import FusionModel


def find_latest_checkpoint(ckpt_dir=config.CHECKPOINTS_DIR):
    ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}. Train the model first.")
    return os.path.join(ckpt_dir, ckpts[-1])


def load_models(checkpoint_path, device):
    model   = FusionModel(n_cls=config.N_CLS).to(device)
    img_enc = ImageEncoder(device)
    txt_enc = TextEncoder(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "img_enc" in ckpt: img_enc.load_state_dict(ckpt["img_enc"])
        if "txt_enc" in ckpt: txt_enc.load_state_dict(ckpt["txt_enc"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    img_enc.eval()
    txt_enc.eval()
    return model, img_enc, txt_enc


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return img_rgb, pil_img


def predict(image_path, text="", checkpoint_path=None, device=None):
    """
    Run inference on a single comic panel.

    Returns:
        (label: str, confidence: float, class_probs: dict)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()

    model, img_enc, txt_enc = load_models(checkpoint_path, device)

    img_rgb, pil_img = preprocess_image(image_path)

    with torch.no_grad():
        img_feat = img_enc([pil_img])                         # (1, 512)
        txt_feat = txt_enc([text])                            # (1, 768)

        outputs = model(img_feat, txt_feat)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()

    label      = config.IDX_TO_LABEL[pred_idx]
    confidence = probs[pred_idx].item()
    class_probs = {config.IDX_TO_LABEL[i]: probs[i].item() for i in range(config.N_CLS)}

    return label, confidence, class_probs


def main():
    p = argparse.ArgumentParser(description="Comic Emotion Classification — Inference")
    p.add_argument("image", type=str, help="Path to comic panel image")
    p.add_argument("--text", type=str, default="", help="Optional dialogue text")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint path (defaults to latest in checkpoints/)")
    p.add_argument("--device", type=str, default=None,
                   choices=["cuda", "cpu"],
                   help="Device (auto-detected if not specified)")
    args = p.parse_args()

    # Auto-select device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    ckpt = args.checkpoint or find_latest_checkpoint()

    print(f"\nRunning inference on : {args.image}")
    print(f"Dialogue text        : '{args.text}'")
    print(f"Checkpoint           : {ckpt}")
    print(f"Device               : {device}")
    print("-" * 55)

    try:
        label, conf, probs = predict(args.image, args.text, ckpt, device)

        print(f"\nPredicted Emotion : {label.upper()}")
        print(f"Confidence        : {conf:.4f}")
        print("\nClass Probabilities:")
        for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(prob * 20)
            print(f"  {emotion:10s}: {prob:.4f}  {bar}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
