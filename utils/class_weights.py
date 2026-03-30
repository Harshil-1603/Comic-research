"""
Compute inverse-frequency class weights from a CSV split.
Used to counter class imbalance in CrossEntropyLoss.
"""
import torch
import pandas as pd
import numpy as np
import config


def compute_class_weights(csv_path, split=None, device="cpu"):
    """
    Compute per-class weights = total_samples / (n_classes * class_count).
    Classes that never appear get weight 0 (excluded).

    Returns:
        torch.FloatTensor of shape (N_CLS,) on `device`
    """
    df = pd.read_csv(csv_path)
    if split and "split" in df.columns:
        df = df[df["split"] == split]

    counts = df["emotion"].value_counts()
    total  = len(df)
    n_cls  = config.N_CLS

    weights = torch.zeros(n_cls, dtype=torch.float32)
    for emotion, idx in config.LABEL_MAP.items():
        c = counts.get(emotion, 0)
        if c > 0:
            weights[idx] = total / (n_cls * c)   # inverse frequency

    print("Class weights:")
    for emotion, idx in config.LABEL_MAP.items():
        c = counts.get(emotion, 0)
        print(f"  {emotion:10s}: count={c:3d}  weight={weights[idx]:.3f}")

    return weights.to(device)
