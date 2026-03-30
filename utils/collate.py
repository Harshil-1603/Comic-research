"""
Custom collate function for ComicDataset.
Returns imgs (list of numpy arrays), texts (list of str), and ys (stacked LongTensor).
"""
import torch


def collate(batch):
    imgs, texts, ys = zip(*batch)
    return list(imgs), list(texts), torch.stack(ys)  # (B,) LongTensor
