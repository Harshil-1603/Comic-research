import os, cv2, torch
import pandas as pd
from torch.utils.data import Dataset

label_map = {"anger": 0, "sadness": 1, "joy": 2, "fear": 3, "neutral": 4}


class ComicDataset(Dataset):
    def __init__(self, csv, img_dir, transforms=None):
        self.df = pd.read_csv(csv)
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(os.path.join(self.img_dir, r["image"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        y = label_map[r["emotion"]]
        text = r["text"] if isinstance(r["text"], str) else ""
        return img, text, torch.tensor(y)
