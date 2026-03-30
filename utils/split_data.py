"""
Stratified train/val/test split (70/15/15)
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_data(csv_path, output_dir="data/processed", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split annotations into train/val/test with stratification.
    Default split: 70/15/15
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    df = pd.read_csv(csv_path)
    
    # First split: separate test set
    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df['emotion'], random_state=seed
    )
    
    # Second split: separate train and val
    val_size = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val['emotion'], random_state=seed
    )
    
    # Add split column
    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'
    
    # Combine and save
    combined = pd.concat([train, val, test])
    output_path = os.path.join(output_dir, 'annotations_split.csv')
    combined.to_csv(output_path, index=False)
    
    # Print statistics
    print(f"Data split saved to {output_path}")
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train)} ({len(train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")
    print(f"\nClass distribution per split:")
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()}:")
        split_df = combined[combined['split'] == split_name]
        print(split_df['emotion'].value_counts().sort_index())
    
    return combined


if __name__ == "__main__":
    split_data("data/annotations.csv")
