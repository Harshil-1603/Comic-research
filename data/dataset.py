"""
PyTorch Dataset for comic panel emotion classification.
7-class schema: anger, disgust, fear, joy, neutral, sadness, surprise.
"""
import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import config

label_map = config.LABEL_MAP   # single source of truth


class ComicDataset(Dataset):
    """
    Args:
        csv_path  : annotations CSV (columns: image, emotion, text[, source, split])
        img_dir   : directory containing panel JPGs
        transforms: optional callable applied to the uint8 RGB numpy image
        split     : 'train' | 'val' | 'test' | None
    """

    def __init__(self, csv_path, img_dir, transforms=None, split=None):
        df = pd.read_csv(csv_path)

        if split is not None:
            if "split" not in df.columns:
                raise ValueError("CSV has no 'split' column. Run: python utils/split_data.py")
            df = df[df["split"] == split].reset_index(drop=True)

        # Drop rows whose emotion label is not in the current schema
        valid = set(label_map.keys())
        bad = ~df["emotion"].isin(valid)
        if bad.any():
            print(f"  ⚠  Dropping {bad.sum()} rows with unknown/old emotion labels.")
            df = df[~bad].reset_index(drop=True)

        self.df         = df
        self.img_dir    = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])
        img      = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)

        label = label_map[row["emotion"]]
        text  = row["text"] if isinstance(row["text"], str) else ""

        return img, text, torch.tensor(label)
