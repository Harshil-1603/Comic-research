"""
Stratified train/val/test split (70/15/15) — plan §17.

Output: data/annotations_split.csv  (adds a 'split' column)

Usage:
    python utils/split_data.py
    python utils/split_data.py --csv data/annotations.csv
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def split_data(csv_path=config.ANNOTATIONS_CSV,
               output_path=config.SPLIT_CSV,
               train_ratio=config.TRAIN_RATIO,
               val_ratio=config.VAL_RATIO,
               test_ratio=config.TEST_RATIO,
               seed=config.SEED):
    """
    Split annotations into train/val/test with stratification.
    Writes a new CSV with an added 'split' column.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError("annotations.csv is empty — annotate some panels first.")

    # Validate emotions
    valid = set(config.LABEL_MAP.keys())
    bad = df[~df["emotion"].isin(valid)]
    if len(bad):
        raise ValueError(f"Unknown emotion labels: {bad['emotion'].unique()}")

    # Split: test out first
    try:
        train_val, test = train_test_split(
            df, test_size=test_ratio, stratify=df["emotion"], random_state=seed
        )
    except ValueError:
        print("  ⚠  Warning: Too few samples for stratified test split. Falling back to random split.")
        train_val, test = train_test_split(
            df, test_size=test_ratio, random_state=seed
        )

    # Split: val from train_val
    val_size = val_ratio / (train_ratio + val_ratio)
    try:
        train, val = train_test_split(
            train_val, test_size=val_size, stratify=train_val["emotion"], random_state=seed
        )
    except ValueError:
        print("  ⚠  Warning: Too few samples for stratified val split. Falling back to random split.")
        train, val = train_test_split(
            train_val, test_size=val_size, random_state=seed
        )

    train = train.copy(); train["split"] = "train"
    val   = val.copy();   val["split"]   = "val"
    test  = test.copy();  test["split"]  = "test"

    combined = pd.concat([train, val, test]).reset_index(drop=True)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.to_csv(output_path, index=False)

    n = len(df)
    print(f"Split CSV saved to {output_path}")
    print(f"\nSplit sizes:")
    print(f"  Train : {len(train):4d}  ({len(train)/n*100:.1f}%)")
    print(f"  Val   : {len(val):4d}  ({len(val)/n*100:.1f}%)")
    print(f"  Test  : {len(test):4d}  ({len(test)/n*100:.1f}%)")
    print("\nClass distribution per split:")
    for split_name in ["train", "val", "test"]:
        sub = combined[combined["split"] == split_name]
        print(f"\n{split_name.upper()}:")
        print(sub["emotion"].value_counts().sort_index().to_string())

    return combined


def main():
    p = argparse.ArgumentParser(description="Create stratified train/val/test split")
    p.add_argument("--csv", type=str, default=config.ANNOTATIONS_CSV)
    p.add_argument("--output", type=str, default=config.SPLIT_CSV)
    args = p.parse_args()
    split_data(args.csv, args.output)


if __name__ == "__main__":
    main()
