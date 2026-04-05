import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from data.dataset import ComicDataset
from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils.collate import collate

tfm = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
])

def to_pil(batch_imgs):
    return [tfm(img) for img in batch_imgs]

def extract_for_split(csv_path, split, img_enc, txt_enc, device):
    try:
        ds = ComicDataset(csv_path, config.PROCESSED_DIR, split=split)
    except Exception as e:
        return []

    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate, num_workers=config.NUM_WORKERS)
    
    img_feats = []
    txt_feats = []
    labels = []

    with torch.no_grad():
        for imgs, texts, ys in tqdm(loader, desc=f"Extracting {split}"):
            pil_imgs = to_pil(imgs)
            with torch.amp.autocast('cuda', enabled=True):
                feat_i = img_enc(pil_imgs).cpu()
                feat_t = txt_enc(texts).cpu()
            
            img_feats.append(feat_i)
            txt_feats.append(feat_t)
            labels.append(ys.cpu())
            
    if img_feats:
        return {
            "img":   torch.cat(img_feats, dim=0),
            "txt":   torch.cat(txt_feats, dim=0),
            "label": torch.cat(labels, dim=0)
        }
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device

    print(f"Loading encoders on {device}...")
    img_enc = ImageEncoder(device).eval()
    txt_enc = TextEncoder(device).eval()
    
    csv_path = config.SPLIT_CSV if os.path.exists(config.SPLIT_CSV) else config.ANNOTATIONS_CSV

    print(f"Extracting features using {csv_path}...")
    output = {}
    
    # Extract features for all splits natively
    for split in ["train", "val", "test", None]:
        if split is None and output:  # We already found splits
            break
        res = extract_for_split(csv_path, split, img_enc, txt_enc, device)
        if res:
            split_key = split if split else "all"
            output[split_key] = res

    out_path = "data/embeddings.pt"
    torch.save(output, out_path)
    print(f"\nSaved embeddings to {out_path}")
    print("Done.\n")

if __name__ == "__main__":
    main()
