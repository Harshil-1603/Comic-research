import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, pt_path, split="train"):
        data = torch.load(pt_path, map_location="cpu")
        # Handle if no splits were generated
        if split not in data and "all" in data:
            self.subset = data["all"]
        else:
            self.subset = data[split]

        self.img   = self.subset["img"]
        self.txt   = self.subset["txt"]
        self.label = self.subset["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.img[idx], self.txt[idx], self.label[idx]
